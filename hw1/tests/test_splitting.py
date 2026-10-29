"""Check temporal boundaries, tied events, and frozen-protocol integrity."""

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

from utils.reproducibility import load_config
from utils.splitting import (
    choose_temporal_boundaries,
    save_split_report,
    split_content_sha256,
    temporal_split,
)


def events(days):
    count = len(days)
    return pd.DataFrame({
        "user_id": pd.Series([f"synthetic-user-{i}" for i in range(count)], dtype="string"),
        "item_id": pd.Series([f"synthetic-item-{i}" for i in range(count)], dtype="string"),
        "rating": [5.0] * count,
        "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(days, unit="D"),
        "source_line": range(1, count + 1),
    })


class TemporalSplitTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.frame = events(list(range(10)))
        self.config["split"].update({
            "validation_start_utc": "2020-01-09T00:00:00+00:00",
            "test_start_utc": "2020-01-10T00:00:00+00:00",
            "boundary_source_content_sha256": None,
        })

    def test_unique_timestamps_produce_eighty_ten_ten(self):
        t1, t2 = choose_temporal_boundaries(self.frame["timestamp"], [0.8, 0.9])
        self.assertEqual(t1, pd.Timestamp("2020-01-09", tz="UTC"))
        self.assertEqual(t2, pd.Timestamp("2020-01-10", tz="UTC"))
        parts, report = temporal_split(self.frame, self.config)
        self.assertEqual([len(part) for part in parts.values()], [8, 1, 1])
        self.assertTrue(all(report["checks"].values()))

    def test_tied_quantiles_still_produce_three_nonempty_periods(self):
        frame = events([0] * 98 + [1, 2])
        t1, t2 = choose_temporal_boundaries(frame["timestamp"], [0.8, 0.9])
        self.config["split"].update({"validation_start_utc": t1.isoformat(), "test_start_utc": t2.isoformat()})
        parts, report = temporal_split(frame, self.config)
        self.assertEqual([len(part) for part in parts.values()], [98, 1, 1])
        self.assertTrue(report["checks"]["equal_timestamps_stay_together"])

    def test_events_on_cutoffs_enter_validation_and_test(self):
        frame = events([0, 8, 8, 9, 9])
        parts, _ = temporal_split(frame, self.config)
        self.assertEqual(parts["train"]["source_line"].tolist(), [1])
        self.assertEqual(parts["validation"]["source_line"].tolist(), [2, 3])
        self.assertEqual(parts["test"]["source_line"].tolist(), [4, 5])

    def test_event_counts_not_unique_dates_determine_the_boundary(self):
        timestamps = events([0] * 8 + [1, 2])["timestamp"]
        t1, t2 = choose_temporal_boundaries(timestamps, [0.8, 0.9])
        self.assertEqual(t1, pd.Timestamp("2020-01-02", tz="UTC"))
        self.assertEqual(t2, pd.Timestamp("2020-01-03", tz="UTC"))

    def test_ties_in_boundary_distance_choose_the_earlier_time(self):
        t1, _ = choose_temporal_boundaries(events([0, 1, 2, 3])["timestamp"], [0.375, 0.875])
        self.assertEqual(t1, pd.Timestamp("2020-01-02", tz="UTC"))

    def test_cutoffs_and_membership_are_independent_of_rating_values(self):
        modified = self.frame.copy()
        modified["rating"] = [1.0, 5.0] * 5
        self.assertEqual(
            choose_temporal_boundaries(self.frame["timestamp"], [0.8, 0.9]),
            choose_temporal_boundaries(modified["timestamp"], [0.8, 0.9]),
        )
        _, first = temporal_split(self.frame, self.config)
        _, second = temporal_split(modified, self.config)
        self.assertEqual(first["membership_sha256"], second["membership_sha256"])
        self.assertEqual(first["periods"], second["periods"])

    def test_permutation_does_not_change_membership_or_report(self):
        first, first_report = temporal_split(self.frame, self.config)
        second, second_report = temporal_split(self.frame.sample(frac=1, random_state=42), self.config)
        self.assertEqual(first_report, second_report)
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_invalid_quantiles_or_too_few_timestamp_groups_fail(self):
        for quantiles in ([0.9, 0.8], [0, 0.9], [0.8, 1], [0.8]):
            with self.subTest(quantiles=quantiles):
                with self.assertRaisesRegex(ValueError, "quantiles"):
                    choose_temporal_boundaries(self.frame["timestamp"], quantiles)
        with self.assertRaisesRegex(ValueError, "three distinct"):
            choose_temporal_boundaries(events([0, 0, 1])["timestamp"], [0.8, 0.9])

    def test_unfixed_reversed_naive_or_empty_period_cutoffs_fail(self):
        for first, second in (
            (None, None),
            ("2020-01-10T00:00:00+00:00", "2020-01-09T00:00:00+00:00"),
            ("2020-01-09", "2020-01-10"),
            ("2010-01-01T00:00:00+00:00", "2011-01-01T00:00:00+00:00"),
        ):
            with self.subTest(first=first, second=second):
                config = deepcopy(self.config)
                config["split"].update({"validation_start_utc": first, "test_start_utc": second})
                with self.assertRaises(ValueError):
                    temporal_split(self.frame, config)

    def test_naive_or_missing_timestamps_fail(self):
        naive = self.frame["timestamp"].dt.tz_localize(None)
        with self.assertRaisesRegex(ValueError, "UTC"):
            choose_temporal_boundaries(naive, [0.8, 0.9])
        missing = self.frame["timestamp"].copy()
        missing.iloc[0] = pd.NaT
        with self.assertRaisesRegex(ValueError, "non-null"):
            choose_temporal_boundaries(missing, [0.8, 0.9])

    def test_frozen_dataset_hash_rejects_changed_input(self):
        self.config["split"]["boundary_source_content_sha256"] = split_content_sha256(self.frame)
        temporal_split(self.frame, self.config)
        modified = self.frame.copy()
        modified.loc[0, "rating"] = 1
        with self.assertRaisesRegex(ValueError, "differs from the dataset"):
            temporal_split(modified, self.config)

    def test_repeated_source_lines_fail(self):
        self.frame.loc[1, "source_line"] = 1
        with self.assertRaisesRegex(ValueError, "Source lines"):
            temporal_split(self.frame, self.config)

    def test_only_local_file_contains_row_membership(self):
        parts, report = temporal_split(self.frame, self.config)
        with tempfile.TemporaryDirectory() as directory:
            self.config["storage"]["results_dir"] = str(Path(directory) / "results")
            with patch.dict(os.environ, {"RECSYS_DATA_DIR": directory}):
                path = save_split_report(parts, report, self.config)
                membership = gzip.decompress(path.read_bytes())
                self.assertEqual(hashlib.sha256(membership).hexdigest(), report["membership_sha256"])
                self.assertEqual(len(pd.read_csv(path)), len(self.frame))
            published = json.loads((Path(directory) / "results/split_summary.json").read_text())
            self.assertEqual(published, report)
            self.assertFalse(published["held_out_rating_statistics_included"])
            self.assertNotIn("synthetic-user", json.dumps(published))


if __name__ == "__main__":
    unittest.main()
