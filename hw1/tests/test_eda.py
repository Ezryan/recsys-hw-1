"""Verify descriptive sparsity and strict historical metadata on synthetic data."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from utils.eda import build_eda_profile, save_eda_tables
from utils.reproducibility import load_config


class EdaTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["split"]["validation_start_utc"] = "2020-01-06T00:00:00+00:00"
        self.frame = pd.DataFrame({
            "user_id": ["synthetic-user-A"] * 3 + ["synthetic-user-B"],
            "item_id": ["synthetic-item-A", "synthetic-item-B", "synthetic-item-C", "synthetic-item-A"],
            "rating": [5., 1., 5., 3.],
            "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta([0, 1, 1, 2], unit="D"),
            "source_line": [1, 2, 3, 4],
            "summary": [None, None, "synthetic", None],
        })

    def test_sparsity_uses_the_same_all_user_item_universe(self):
        profile = build_eda_profile(self.frame, self.config)
        table = profile["tables"]["sparsity"].set_index("signal")
        self.assertEqual(table["possible_pairs"].tolist(), [6, 6])
        self.assertEqual(table["nnz"].tolist(), [4, 2])
        self.assertAlmostEqual(table.loc["all_reviews", "sparsity"], 1/3)
        self.assertAlmostEqual(table.loc["positive_reviews", "sparsity"], 2/3)

    def test_metadata_excludes_the_entire_current_timestamp_group(self):
        profile = build_eda_profile(self.frame, self.config)
        users = profile["event_metadata"]["user"]
        items = profile["event_metadata"]["item"]
        self.assertEqual(users["mean_rating"].tolist(), [3., 5., 5., 3.])
        self.assertEqual(users["positive_fraction"].tolist(), [0., 1., 1., 0.])
        self.assertEqual(users["empty_history"].tolist(), [1, 0, 0, 1])
        self.assertTrue(np.allclose(users["log1p_review_count"], np.log1p([0, 1, 1, 0])))
        self.assertAlmostEqual(items.loc[3, "log1p_observed_age_days"], np.log1p(2))
        self.assertEqual(profile["summary"]["training_events_without_prior_item_history"], 3)
        self.assertEqual(profile["summary"]["training_events_without_either_history"], 1)

    def test_current_or_future_ratings_cannot_change_earlier_features(self):
        before = build_eda_profile(self.frame, self.config)
        modified = self.frame.copy()
        modified.loc[1, "rating"] = 4.
        modified.loc[3, "rating"] = 1.
        after = build_eda_profile(modified, self.config)
        for entity in ("user", "item"):
            pd.testing.assert_frame_equal(
                before["event_metadata"][entity].iloc[:3], after["event_metadata"][entity].iloc[:3]
            )

    def test_snapshot_and_fallback_metadata_are_explicit(self):
        profile = build_eda_profile(self.frame, self.config)
        snapshot = profile["snapshot_metadata"]["user"]
        self.assertAlmostEqual(snapshot.loc["synthetic-user-A", "mean_rating"], 11/3)
        self.assertEqual(snapshot["empty_history"].tolist(), [0, 0])
        self.assertEqual(profile["summary"]["metadata_missing_values_after_defaults"], 0)
        self.assertEqual(profile["summary"]["n_users_without_positive_history"], 1)

    def test_future_empty_or_repeated_pair_input_is_rejected(self):
        future = self.frame.copy()
        future.loc[3, "timestamp"] = pd.Timestamp(self.config["split"]["validation_start_utc"])
        with self.assertRaisesRegex(ValueError, "strictly before validation"):
            build_eda_profile(future, self.config)
        with self.assertRaises(ValueError):
            build_eda_profile(self.frame.iloc[:0], self.config)
        with self.assertRaisesRegex(ValueError, "unique user-item"):
            build_eda_profile(pd.concat([self.frame, self.frame.iloc[:1]]), self.config)

    def test_no_positive_reviews_still_preserves_zero_histories(self):
        frame = self.frame.copy()
        frame["rating"] = 1.
        profile = build_eda_profile(frame, self.config)
        self.assertEqual(profile["summary"]["n_positive_interactions"], 0)
        self.assertEqual(profile["summary"]["n_users_without_positive_history"], 2)
        self.assertEqual(profile["summary"]["sparsity_positive_reviews"], 1.)

    def test_input_is_unchanged_and_permutation_preserves_features(self):
        original = self.frame.copy(deep=True)
        first = build_eda_profile(self.frame, self.config)
        second = build_eda_profile(self.frame.sample(frac=1, random_state=42), self.config)
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(first["summary"], second["summary"])
        for entity in ("user", "item"):
            pd.testing.assert_frame_equal(
                first["event_metadata"][entity].sort_index(), second["event_metadata"][entity].sort_index()
            )

    def test_published_reports_have_no_individual_records_or_ids(self):
        profile = build_eda_profile(self.frame, self.config)
        with tempfile.TemporaryDirectory() as directory:
            self.config["storage"]["results_dir"] = directory
            save_eda_tables(profile, self.config)
            summary = json.loads((Path(directory) / "eda_summary.json").read_text())
            self.assertFalse(summary["test_rating_statistics_included"])
            for path in Path(directory).iterdir():
                content = path.read_text()
                self.assertNotIn("synthetic-user-A", content)
                self.assertNotIn("synthetic-item-A", content)


if __name__ == "__main__":
    unittest.main()
