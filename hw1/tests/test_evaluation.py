"""Synthetic tests of shared candidates, cold start, and exact batched top-k."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from utils.evaluation import (
    build_evaluation_stage, build_history_context, recommend_top_k,
    save_evaluation_stage, validate_recommendations,
)
from utils.reproducibility import load_config


def interactions(rows, date):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "source_line"]).assign(
        timestamp=pd.Timestamp(date, tz="UTC")
    )


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["split"].update({
            "validation_start_utc": "2020-01-03T00:00:00+00:00",
            "test_start_utc": "2020-01-05T00:00:00+00:00",
        })
        history = [("user-A", "item-00", 5, 1), ("user-A", "item-01", 1, 2)]
        history += [("user-B", f"item-{i:02d}", 5, i + 2) for i in range(1, 14)]
        history += [("user-Low", "item-00", 1, 16), ("user-Low", "item-14", 1, 17)]
        validation = [
            ("user-A", "item-02", 5, 100), ("user-A", "item-02", 4, 101),
            ("user-A", "item-00", 5, 102), ("user-A", "item-01", 5, 103),
            ("user-A", "item-14", 5, 104), ("user-A", "item-99", 5, 105),
            ("user-New", "item-03", 5, 106), ("user-Low", "item-03", 5, 107),
            ("user-B", "item-00", 5, 108), ("user-A", "item-03", 1, 109),
        ]
        # No test frame is supplied: validation preparation must not access it.
        self.periods = {
            "train": interactions(history, "2020-01-01"),
            "validation": interactions(validation, "2020-01-03"),
        }

    def stage(self):
        return build_evaluation_stage(self.periods, self.config)

    def test_history_vocabulary_and_masks_include_low_ratings(self):
        context = self.stage().context
        a = context.user_ids.index("user-A")
        low = context.user_ids.index("user-Low")
        item = context.item_ids.index("item-01")
        self.assertEqual(context.user_ids, tuple(sorted(context.user_ids)))
        self.assertEqual(context.item_ids, tuple(sorted(context.item_ids)))
        self.assertTrue(context.seen[a, item])
        self.assertEqual(context.positive[a, item], 0)
        self.assertFalse(context.warm_user_mask[low])
        self.assertFalse(context.catalog_mask[context.item_ids.index("item-14")])
        self.assertEqual(context.candidate_counts[a], 12)
        self.assertEqual(context.positive.dtype, np.float32)

    def test_ground_truth_is_a_set_and_cohort_is_common(self):
        stage = self.stage()
        self.assertEqual([stage.context.user_ids[i] for i in stage.user_indices], ["user-A"])
        self.assertEqual(stage.ground_truth.nnz, 1)
        self.assertEqual(stage.ground_truth[stage.user_indices[0], stage.context.item_ids.index("item-02")], 1)
        self.assertEqual(stage.report["held_out_positive_events"], 9)
        self.assertEqual(stage.report["positive_event_audit"][-1]["remaining_positive_events"], 2)
        self.assertEqual(stage.report["n_warm_users_with_too_few_candidates"], 1)
        self.assertFalse(stage.report["test_labels_used"])

    def test_unknown_ids_are_not_aliased_to_last_index(self):
        stage = self.stage()
        cold = stage.report["cold_start"]
        self.assertEqual(cold["new_user_ids"], 1)
        self.assertEqual(cold["known_users_without_positive_history"], 1)
        self.assertEqual(cold["new_item_ids"], 1)
        self.assertEqual(cold["known_items_without_positive_history"], 1)
        self.assertEqual(cold["positive_events_with_either"], 4)
        self.assertEqual(stage.ground_truth[stage.context.user_ids.index("user-Low")].nnz, 0)
        self.assertNotIn("item-99", stage.context.item_ids)

    def test_zero_score_ties_follow_original_ids_across_block_boundaries(self):
        stage = self.stage()
        expected = [stage.context.item_ids.index(f"item-{i:02d}") for i in range(2, 12)]
        for block_size in (1, 2, 3, 7, 100):
            with self.subTest(block=block_size):
                result = recommend_top_k(lambda u, i: np.zeros((len(u), len(i))), stage.context,
                                         stage.user_indices, item_batch_size=block_size)
                self.assertEqual(result.tolist(), [expected])
                self.assertTrue(all(validate_recommendations(result, stage).values()))
        # item-03 has a future negative event, but is still a candidate at cutoff.
        self.assertIn(stage.context.item_ids.index("item-03"), expected)

    def test_batched_ranker_matches_dense_lexicographic_reference(self):
        self.periods["train"] = pd.concat([
            self.periods["train"], interactions([("user-C", "item-00", 5, 18)], "2020-01-01")
        ])
        self.periods["validation"] = pd.concat([
            self.periods["validation"], interactions([("user-C", "item-04", 5, 110)], "2020-01-03")
        ])
        stage = self.stage()
        context = stage.context
        scores = np.random.default_rng(42).integers(-2, 3, size=context.positive.shape).astype(float)
        original = scores.copy()
        expected = []
        for user in stage.user_indices:
            candidates = context.catalog_items[~context.seen[user, context.catalog_items].toarray().ravel()]
            order = np.lexsort((candidates, -scores[user, candidates]))
            expected.append(candidates[order[:10]])
        for user_batch, item_batch in ((1, 2), (2, 3), (10, 100)):
            result = recommend_top_k(lambda u, i: scores[np.ix_(u, i)], context, stage.user_indices,
                                     user_batch_size=user_batch, item_batch_size=item_batch)
            np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(scores, original)

    def test_ranker_rejects_unknown_users_short_candidates_and_bad_scores(self):
        stage = self.stage()
        context = stage.context
        zero = lambda u, i: np.zeros((len(u), len(i)))
        for users in (np.array([-1]), np.array([len(context.user_ids)]),
                      np.array([context.user_ids.index("user-B")]), np.repeat(stage.user_indices, 2)):
            with self.subTest(users=users):
                with self.assertRaises(ValueError):
                    recommend_top_k(zero, context, users)
        for scorer in (lambda u, i: np.full((len(u), len(i)), np.nan), lambda u, i: np.zeros(1)):
            with self.assertRaisesRegex(ValueError, "finite user-by-item"):
                recommend_top_k(scorer, context, stage.user_indices)

    def test_recommendation_validation_rejects_duplicates_seen_and_outside_catalog(self):
        stage = self.stage()
        valid = recommend_top_k(lambda u, i: np.zeros((len(u), len(i))), stage.context, stage.user_indices)
        for replacement in (valid[0, 1], stage.context.item_ids.index("item-01"),
                            stage.context.item_ids.index("item-14"), -1):
            invalid = valid.copy()
            invalid[0, 0] = replacement
            with self.subTest(item=replacement):
                with self.assertRaises(ValueError):
                    validate_recommendations(invalid, stage)
        with self.assertRaises(ValueError):
            validate_recommendations(valid[:, :9], stage)

    def test_permutation_preserves_vocabulary_context_and_cohort_hashes(self):
        first = self.stage()
        for period in self.periods:
            self.periods[period] = self.periods[period].sample(frac=1, random_state=123)
        second = self.stage()
        self.assertEqual(first.context.user_ids, second.context.user_ids)
        self.assertEqual(first.context.item_ids, second.context.item_ids)
        self.assertEqual(first.report, second.report)

    def test_test_refit_expands_history_and_excludes_validation_seen_items(self):
        self.periods["validation"] = interactions([
            ("user-A", "item-02", 5, 100), ("user-New", "item-03", 5, 101)
        ], "2020-01-03")
        self.periods["test"] = interactions([
            ("user-New", "item-04", 5, 200), ("user-A", "item-02", 5, 201),
            ("user-A", "item-04", 5, 202)
        ], "2020-01-05")
        stage = build_evaluation_stage(self.periods, self.config, "test")
        a = stage.context.user_ids.index("user-A")
        new = stage.context.user_ids.index("user-New")
        self.assertTrue(stage.context.warm_user_mask[new])
        self.assertTrue(stage.context.seen[a, stage.context.item_ids.index("item-02")])
        self.assertEqual(stage.ground_truth[a, stage.context.item_ids.index("item-02")], 0)
        self.assertEqual(stage.ground_truth.nnz, 2)

    def test_temporal_leakage_and_overlapping_source_lines_are_rejected(self):
        for column, value in (("timestamp", pd.Timestamp("2020-01-02", tz="UTC")), ("source_line", 1)):
            with self.subTest(column=column):
                periods = {key: value.copy() for key, value in self.periods.items()}
                periods["validation"].iloc[0, periods["validation"].columns.get_loc(column)] = value
                with self.assertRaises(ValueError):
                    build_evaluation_stage(periods, self.config)
        with self.assertRaisesRegex(ValueError, "strictly before"):
            build_history_context(self.periods["train"], "2019-01-01T00:00:00+00:00", self.config)

    def test_validation_never_accesses_test_frame(self):
        first = self.stage()
        self.periods["test"] = object()
        second = self.stage()
        self.assertEqual(first.report["stage_sha256"], second.report["stage_sha256"])

    def test_cached_matrices_are_local_and_public_report_has_no_ids(self):
        stage = self.stage()
        with tempfile.TemporaryDirectory() as directory:
            self.config["storage"]["results_dir"] = str(Path(directory) / "results")
            with patch.dict("os.environ", {"RECSYS_DATA_DIR": directory}):
                output = save_evaluation_stage(stage, self.config)
                restored = sparse.load_npz(output / "history_seen.npz")
                self.assertEqual((restored != stage.context.seen).nnz, 0)
                mapping = json.loads((output / "mappings.json").read_text())
                self.assertEqual(mapping["user_ids"], list(stage.context.user_ids))
            public = (Path(directory) / "results/evaluation_validation_summary.json").read_text()
            self.assertNotIn("user-A", public)
            self.assertNotIn("item-02", public)


if __name__ == "__main__":
    unittest.main()
