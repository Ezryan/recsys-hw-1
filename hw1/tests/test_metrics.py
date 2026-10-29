"""Analytical ranking metric checks independent of any recommender model."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from utils.evaluation import (
    build_evaluation_stage, evaluate_recommendations, evaluate_scorer, metric_sanity_checks,
)
from utils.reproducibility import load_config


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["split"].update({
            "validation_start_utc": "2020-01-02T00:00:00+00:00",
            "test_start_utc": "2020-01-03T00:00:00+00:00",
        })
        self.config["runtime"].update({"user_batch_size": 1, "item_batch_size": 3})

    def stage_for(self, relevant_sets):
        history = [("seed", f"candidate-{i:03d}", 5., i + 1) for i in range(50)]
        history += [(f"user-{u:03d}", "candidate-050", 5., 100 + u) for u in range(len(relevant_sets))]
        future = [(f"user-{u:03d}", f"candidate-{i:03d}", 5., 1000 + u * 100 + i)
                  for u, relevant in enumerate(relevant_sets) for i in sorted(relevant)]
        columns = ["user_id", "item_id", "rating", "source_line"]
        periods = {
            "train": pd.DataFrame(history, columns=columns).assign(timestamp=pd.Timestamp("2020-01-01", tz="UTC")),
            "validation": pd.DataFrame(future, columns=columns).assign(timestamp=pd.Timestamp("2020-01-02", tz="UTC")),
        }
        return build_evaluation_stage(periods, self.config)

    def recommendations(self, stage, rows):
        mapping = {value: i for i, value in enumerate(stage.context.item_ids)}
        return np.array([[mapping[f"candidate-{i:03d}"] for i in row] for row in rows], dtype=np.int64)

    def test_perfect_ranking_with_fewer_than_ten_relevant_items(self):
        stage = self.stage_for([{0, 1, 2}])
        result = evaluate_recommendations(self.recommendations(stage, [list(range(10))]), stage)
        self.assertEqual(result.summary["ndcg@10"], 1.)
        self.assertEqual(result.summary["recall@10"], 1.)

    def test_no_hits_are_zero_and_still_count_in_macro_average(self):
        stage = self.stage_for([{20, 21}])
        result = evaluate_recommendations(self.recommendations(stage, [list(range(10))]), stage)
        self.assertEqual(result.summary["ndcg@10"], 0.)
        self.assertEqual(result.summary["recall@10"], 0.)
        self.assertEqual(result.summary["n_users"], 1)

    def test_moving_a_hit_to_rank_ten_lowers_only_ndcg(self):
        stage = self.stage_for([{0}])
        first = evaluate_recommendations(self.recommendations(stage, [list(range(10))]), stage)
        last = evaluate_recommendations(self.recommendations(stage, [list(range(1, 10)) + [0]]), stage)
        self.assertAlmostEqual(last.summary["ndcg@10"], 1 / math.log2(11))
        self.assertLess(last.summary["ndcg@10"], first.summary["ndcg@10"])
        self.assertEqual(first.summary["recall@10"], last.summary["recall@10"])

    def test_recall_uses_all_relevant_items_even_when_more_than_ten(self):
        stage = self.stage_for([set(range(20))])
        result = evaluate_recommendations(self.recommendations(stage, [list(range(10))]), stage)
        self.assertEqual(result.summary["recall@10"], .5)
        self.assertEqual(result.summary["ndcg@10"], 1.)
        self.assertEqual(result.per_user["n_relevant"].tolist(), [20])
        self.assertAlmostEqual(result.per_user["idcg@10"].iloc[0], sum(1 / math.log2(i + 1) for i in range(1, 11)))

    def test_two_hits_at_first_and_third_positions_match_hand_calculation(self):
        stage = self.stage_for([{0, 2}])
        result = evaluate_recommendations(self.recommendations(stage, [list(range(10))]), stage)
        self.assertEqual(result.per_user["dcg@10"].iloc[0], 1.5)
        self.assertAlmostEqual(result.summary["ndcg@10"], 1.5 / (1 + 1 / math.log2(3)))

    def test_macro_average_gives_each_user_equal_weight(self):
        stage = self.stage_for([{0}, {0, *range(10, 29)}])
        result = evaluate_recommendations(self.recommendations(stage, [list(range(10)), list(range(10))]), stage)
        self.assertAlmostEqual(result.summary["recall@10"], .525)
        self.assertNotAlmostEqual(result.summary["recall@10"], 2 / 21)
        expected_ndcg = (1 + 1 / sum(1 / math.log2(i + 1) for i in range(1, 11))) / 2
        self.assertAlmostEqual(result.summary["ndcg@10"], expected_ndcg)
        np.testing.assert_array_equal(result.per_user["user_index"], stage.user_indices)

    def test_empty_cohorts_or_users_without_relevant_items_are_not_silently_dropped(self):
        empty = self.stage_for([set()])
        with self.assertRaisesRegex(ValueError, "empty evaluation cohort"):
            evaluate_recommendations(np.empty((0, 10), dtype=np.int64), empty)
        stage = self.stage_for([{0}])
        stage.ground_truth = sparse.csr_matrix(stage.ground_truth.shape, dtype=bool)
        with self.assertRaisesRegex(ValueError, "positive number of relevant"):
            evaluate_recommendations(self.recommendations(stage, [list(range(10))]), stage)

    def test_invalid_predictions_or_nonbinary_seen_ground_truth_are_rejected(self):
        stage = self.stage_for([{0}])
        recommendations = self.recommendations(stage, [list(range(10))])
        duplicate = recommendations.copy()
        duplicate[0, 0] = duplicate[0, 1]
        with self.assertRaises(ValueError):
            evaluate_recommendations(duplicate, stage)
        binary = stage.ground_truth.copy()
        stage.ground_truth = binary.astype(float)
        stage.ground_truth.data[:] = 5.
        with self.assertRaisesRegex(ValueError, "binary"):
            evaluate_recommendations(recommendations, stage)
        seen_item = stage.context.item_ids.index("candidate-050")
        stage.ground_truth = sparse.csr_matrix(
            ([True], (stage.user_indices, [seen_item])), shape=stage.context.positive.shape
        )
        with self.assertRaisesRegex(ValueError, "already known"):
            evaluate_recommendations(recommendations, stage)

    def test_common_evaluator_keeps_the_same_users_for_different_scorers(self):
        stage = self.stage_for([{0, 1}, {20}])
        scorers = [
            lambda u, i: np.zeros((len(u), len(i))),
            lambda u, i: np.ones((len(u), len(i))),
            lambda u, i: np.broadcast_to(i, (len(u), len(i))),
            lambda u, i: np.broadcast_to(-i, (len(u), len(i))),
            lambda u, i: np.broadcast_to(u[:, None], (len(u), len(i))),
        ]
        for scorer in scorers:
            result = evaluate_scorer(scorer, stage, self.config)
            self.assertEqual(result.summary["n_users"], 2)
            self.assertEqual(result.summary["stage_sha256"], stage.report["stage_sha256"])
            np.testing.assert_array_equal(result.per_user["user_index"], stage.user_indices)
            self.assertTrue(0 <= result.summary["ndcg@10"] <= 1)
            self.assertTrue(0 <= result.summary["recall@10"] <= 1)

    def test_metrics_do_not_materialize_the_full_ground_truth_matrix(self):
        stage = self.stage_for([{0, 1, 2}])
        with patch.object(stage.ground_truth, "toarray", side_effect=AssertionError("Full dense ground truth")):
            result = evaluate_recommendations(self.recommendations(stage, [list(range(10))]), stage)
        self.assertEqual(result.summary["ndcg@10"], 1.)

    def test_published_sanity_examples_match_their_analytic_answers(self):
        before = deepcopy(self.config)
        table = metric_sanity_checks(self.config)
        self.assertEqual(len(table), 8)
        self.assertTrue(table["passed"].all())
        self.assertEqual(self.config, before)


if __name__ == "__main__":
    unittest.main()
