"""Synthetic checks for cleaning order, first-event selection, and size gates."""

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from utils.cleaning import CORE_COLUMNS, clean_reviews, save_cleaned_reviews, validate_cleaned_reviews
from utils.reproducibility import load_config


def review(line, user="user-1", item="item-1", rating=5, timestamp=1000, **metadata):
    return {
        "source_line": line, "reviewerID": user, "asin": item,
        "overall": rating, "unixReviewTime": timestamp, **metadata,
    }


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        # Deliberately small limits for synthetic unit fixtures only.
        self.config["dataset"]["constraints"].update({
            "min_interactions": 1, "min_users": 1, "min_items": 1,
        })

    def test_invalid_rows_have_separate_sequential_counts(self):
        raw = pd.DataFrame([
            review(1),
            review(2, user=None),
            review(3, item="  "),
            review(4, timestamp="not-a-time"),
            review(5, timestamp=float("inf")),
            review(6, timestamp=False),
            review(7, rating=None),
            review(8, rating=6),
            review(9, rating=float("inf")),
            review(10, rating=True),
            review(11, timestamp=1e300),
        ])
        original = raw.copy(deep=True)
        cleaned, report = clean_reviews(raw, self.config)
        removed = {row["stage"]: row["removed_rows"] for row in report["audit"]}
        self.assertEqual(cleaned["source_line"].tolist(), [1])
        self.assertEqual(removed["drop_missing_ids"], 2)
        self.assertEqual(removed["drop_invalid_timestamps"], 4)
        self.assertEqual(removed["drop_invalid_ratings"], 4)
        self.assertEqual(sum(removed.values()), len(raw) - len(cleaned))
        pd.testing.assert_frame_equal(raw, original)

    def test_nested_exact_duplicates_ignore_dict_key_order(self):
        raw = pd.DataFrame([
            review(9, style={"format": "MP3", "other": [1, {"x": 2}]}),
            review(2, style={"other": [1, {"x": 2}], "format": "MP3"}),
        ])
        cleaned, report = clean_reviews(raw, self.config)
        removed = {row["stage"]: row["removed_rows"] for row in report["audit"]}
        self.assertEqual(removed["drop_exact_duplicates"], 1)
        self.assertEqual(removed["keep_earliest_pair"], 0)
        self.assertEqual(cleaned["source_line"].tolist(), [2])

    def test_different_metadata_is_not_an_exact_duplicate(self):
        raw = pd.DataFrame([
            review(1, style={"formats": ["A", "B"]}),
            review(2, style={"formats": ["B", "A"]}),
        ])
        _, report = clean_reviews(raw, self.config)
        removed = {row["stage"]: row["removed_rows"] for row in report["audit"]}
        self.assertEqual(removed["drop_exact_duplicates"], 0)
        self.assertEqual(removed["keep_earliest_pair"], 1)

    def test_earliest_rating_is_not_replaced_by_a_later_higher_rating(self):
        raw = pd.DataFrame([
            review(1, rating=5, timestamp=2000),
            review(2, rating=1, timestamp=1000),
        ])
        cleaned, report = clean_reviews(raw, self.config)
        self.assertEqual(cleaned["rating"].tolist(), [1.0])
        self.assertEqual(cleaned["source_line"].tolist(), [2])
        self.assertEqual(report["n_low_rating_interactions"], 1)
        self.assertEqual(report["n_positive_interactions"], 0)

    def test_equal_timestamps_use_source_line_without_moving_time(self):
        raw = pd.DataFrame([
            review(3, rating=5),
            review(2, item="item-2"),
            review(1, rating=2),
        ])
        cleaned, _ = clean_reviews(raw, self.config)
        self.assertEqual(cleaned["source_line"].tolist(), [1, 2])
        self.assertEqual(cleaned["timestamp"].nunique(), 1)
        self.assertEqual(cleaned["rating"].tolist(), [2.0, 5.0])
        self.assertEqual(str(cleaned["timestamp"].dt.tz), "UTC")

    def test_string_ids_and_missing_optional_metadata_are_preserved(self):
        raw = pd.DataFrame([
            review(1, user="00042", item="00007", rating=1, summary=None),
            review(2, user="00042", item="00008", summary="synthetic"),
        ])
        cleaned, report = clean_reviews(raw, self.config)
        self.assertEqual(cleaned["user_id"].tolist(), ["00042", "00042"])
        self.assertEqual(cleaned["item_id"].tolist(), ["00007", "00008"])
        self.assertEqual(report["optional_field_missing_rows"]["summary"], 1)
        self.assertEqual(len(cleaned), 2)

    def test_result_does_not_depend_on_dataframe_row_order(self):
        raw = pd.DataFrame([
            review(1, timestamp=2000), review(2, item="item-2"),
            review(3, timestamp=500, rating=2), review(4, item="item-2"),
        ])
        first, first_report = clean_reviews(raw, self.config)
        second, second_report = clean_reviews(raw.sample(frac=1, random_state=123), self.config)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_report, second_report)

    def test_all_assignment_size_limits_are_enforced(self):
        raw = pd.DataFrame([review(1), review(2, user="user-2", item="item-2")])
        for key, value in {"min_interactions": 3, "max_interactions": 1, "min_users": 3, "min_items": 3}.items():
            with self.subTest(limit=key):
                config = deepcopy(self.config)
                config["dataset"]["constraints"][key] = value
                with self.assertRaisesRegex(ValueError, "assignment constraints"):
                    clean_reviews(raw, config)

    def test_original_source_line_must_be_unique(self):
        raw = pd.DataFrame([review(1), review(1, item="item-2")])
        with self.assertRaisesRegex(ValueError, "Source lines"):
            clean_reviews(raw, self.config)

    def test_output_validator_rejects_repeated_pairs_and_naive_time(self):
        cleaned, _ = clean_reviews(pd.DataFrame([review(1), review(2, item="item-2")]), self.config)
        duplicate = cleaned.copy()
        duplicate.loc[1, "item_id"] = "item-1"
        with self.assertRaisesRegex(ValueError, "Repeated"):
            validate_cleaned_reviews(duplicate, self.config)
        naive = cleaned.copy()
        naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "UTC"):
            validate_cleaned_reviews(naive, self.config)

    def test_unimplemented_policy_is_not_silently_ignored(self):
        self.config["cleaning"]["additional_k_core"] = 5
        with self.assertRaisesRegex(ValueError, "Unsupported cleaning policy"):
            clean_reviews(pd.DataFrame([review(1)]), self.config)

    def test_local_csv_is_repeatable_and_public_reports_contain_only_aggregates(self):
        raw = pd.DataFrame([review(1), review(2, item="item-2", rating=2)])
        cleaned, report = clean_reviews(raw, self.config)
        with tempfile.TemporaryDirectory() as directory:
            self.config["storage"]["results_dir"] = str(Path(directory) / "results")
            with patch.dict(os.environ, {"RECSYS_DATA_DIR": directory}):
                path = save_cleaned_reviews(cleaned, report, self.config)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest(), report["content_sha256"])
                save_cleaned_reviews(cleaned, report, self.config)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                restored = pd.read_csv(path, dtype={"user_id": "string", "item_id": "string"}, parse_dates=["timestamp"])
                pd.testing.assert_frame_equal(restored[CORE_COLUMNS], cleaned[CORE_COLUMNS])
            summary = json.loads((Path(directory) / "results/clean_data_summary.json").read_text())
            self.assertTrue(summary["cleaned_constraints_verified"])
            self.assertNotIn("user-1", json.dumps(summary))
            self.assertEqual(list(Path(directory).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
