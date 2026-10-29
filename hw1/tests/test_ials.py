"""Dense normal-equation references and exact-objective checks for iALS."""

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from models import IALS
from models.ials import als_update, ials_objective
from utils.evaluation import build_evaluation_stage
from utils.reproducibility import load_config


def history_frame(rows):
    return pd.DataFrame([
        {"user_id": user, "item_id": item, "rating": float(rating), "source_line": i + 1,
         "timestamp": pd.Timestamp("2020-01-01", tz="UTC")}
        for i, (user, item, rating) in enumerate(rows)
    ])


class IalsTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["split"].update({"validation_start_utc": "2020-01-02T00:00:00+00:00", "test_start_utc": "2020-01-03T00:00:00+00:00"})
        self.observed = sparse.csr_matrix([[1., 0., 1.], [0., 1., 0.]])
        self.y = np.array([[.1, .5], [-.2, .3], [.7, -.1]])
        self.x = np.array([[.3, -.2], [.4, .1]])
        self.history = history_frame([
            ("u0", "a", 5), ("u0", "b", 1), ("u1", "b", 4),
            ("u1", "c", 5), ("u2", "d", 2),
        ])

    def test_one_update_matches_dense_weighted_normal_equations(self):
        targets = self.observed.copy()
        targets.data[:] = [.2, 0., .8]
        for alpha in (0., 2.5):
            actual = als_update(self.observed, targets, self.y, alpha=alpha, regularization=.2)
            confidence = 1 + alpha * self.observed.toarray()
            dense_targets = targets.toarray()
            expected = np.array([
                np.linalg.solve(self.y.T @ np.diag(c) @ self.y + .2 * np.eye(2), self.y.T @ (c * p))
                for c, p in zip(confidence, dense_targets)
            ])
            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_missing_pairs_are_included_in_the_shared_gram(self):
        observed = sparse.csr_matrix([[1., 0., 0.]])
        result = als_update(observed, observed.copy(), np.array([[1.], [2.], [3.]]), alpha=2., regularization=1.)
        self.assertAlmostEqual(result[0, 0], 3 / 17)
        self.assertNotAlmostEqual(result[0, 0], 3 / 4)  # Observed-only regression is different.

    def test_sparse_objective_equals_direct_all_pair_loss(self):
        for values in ([1., 1., 1.], [.2, 0., .8]):
            targets = self.observed.copy()
            targets.data[:] = values
            confidence = 1 + 2.5 * self.observed.toarray()
            expected = np.sum(confidence * np.square(targets.toarray() - self.x @ self.y.T))
            expected += .2 * (np.square(self.x).sum() + np.square(self.y).sum())
            for batch in (1, 100):
                actual = ials_objective(self.observed, targets, self.x, self.y, alpha=2.5, regularization=.2, batch_size=batch)
                self.assertAlmostEqual(actual, expected, places=13)

    def test_every_half_iteration_decreases_the_fixed_training_objective(self):
        model = IALS(rank=2, regularization=.2, alpha=2.5, iterations=4).fit(self.history, self.config)
        values = np.array([row["objective"] for row in model.loss_history_])
        self.assertEqual(len(values), 9)
        self.assertTrue((np.diff(values) <= 1e-10 * np.maximum(1, values[:-1])).all())
        self.assertLess(values[-1], values[0])
        self.assertTrue(model.training_summary_["nonincreasing_objective"])
        user = model.encode_users(["u2"])[0]
        item = model.encode_items(["d"])[0]
        np.testing.assert_array_equal(model.user_factors_[user], np.zeros(2))
        np.testing.assert_array_equal(model.item_factors_[item], np.zeros(2))

    def test_scores_are_factor_products_with_requested_block_order(self):
        model = IALS(rank=3, iterations=2).fit(self.history, self.config)
        users, items = model.encode_users(["u2", "u0"]), model.encode_items(["c", "a", "d"])
        np.testing.assert_allclose(model.score(users, items), model.user_factors_[users] @ model.item_factors_[items].T)

    def test_seed_and_input_permutation_reproduce_the_same_factors(self):
        first = IALS(rank=3, iterations=3).fit(self.history, self.config)
        second = IALS(rank=3, iterations=3).fit(self.history.sample(frac=1, random_state=17), self.config)
        np.testing.assert_array_equal(first.user_factors_, second.user_factors_)
        np.testing.assert_array_equal(first.item_factors_, second.item_factors_)
        self.assertEqual(first.loss_history_, second.loss_history_)
        self.assertEqual(first.training_summary_["factors_sha256"], second.training_summary_["factors_sha256"])

    def test_low_ratings_and_positive_rating_strength_do_not_change_binary_targets(self):
        modified = self.history.copy()
        modified.loc[modified.rating.lt(4), "rating"] = 3.
        modified.loc[modified.rating.ge(4), "rating"] = 4.
        first = IALS(rank=2, iterations=2).fit(self.history, self.config)
        second = IALS(rank=2, iterations=2).fit(modified, self.config)
        np.testing.assert_array_equal(first.user_factors_, second.user_factors_)
        np.testing.assert_array_equal(first.item_factors_, second.item_factors_)
        self.assertEqual(first.observed_.nnz, 3)
        self.assertTrue((first.targets_.data == 1).all())
        self.assertEqual(first.training_summary_["observed_confidence"], 11.)
        self.assertEqual(first.training_summary_["other_confidence"], 1.)

    def test_all_zero_positive_history_has_zero_optimal_factors(self):
        history = history_frame([("u", "a", 1), ("v", "b", 2)])
        model = IALS(rank=2, iterations=2).fit(history, self.config)
        np.testing.assert_array_equal(model.user_factors_, np.zeros((2, 2)))
        np.testing.assert_array_equal(model.item_factors_, np.zeros((2, 2)))
        self.assertEqual(model.loss_history_[-1]["objective"], 0.)

    def test_misaligned_targets_or_nonbinary_observed_data_are_rejected(self):
        targets = self.observed.copy()
        targets.data[0] = 0
        targets.eliminate_zeros()
        with self.assertRaisesRegex(ValueError, "coordinate pattern"):
            als_update(self.observed, targets, self.y, alpha=1., regularization=.2)
        corrupted = self.observed.copy()
        corrupted.data[0] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            als_update(corrupted, self.observed.copy(), self.y, alpha=1., regularization=.2)

    def test_common_stage_recommendations_exclude_all_known_items(self):
        rows = [("A", "item-00", 5), ("A", "item-01", 1)]
        rows += [("seed", f"item-{i:02d}", 5) for i in range(13)]
        history = history_frame(rows)
        future = pd.DataFrame([{"user_id": "A", "item_id": "item-02", "rating": 5., "source_line": 100,
                                "timestamp": pd.Timestamp("2020-01-02", tz="UTC")}])
        stage = build_evaluation_stage({"train": history, "validation": future}, self.config)
        model = IALS(rank=2, iterations=2).fit(history, self.config, context=stage.context)
        recommended = model.decode_items(model.recommend(stage=stage)).ravel().tolist()
        self.assertEqual(len(recommended), 10)
        self.assertEqual(len(set(recommended)), 10)
        self.assertNotIn("item-00", recommended)
        self.assertNotIn("item-01", recommended)
        self.assertEqual(model.evaluate(stage).summary["stage_sha256"], stage.report["stage_sha256"])

    def test_invalid_hyperparameters_are_rejected(self):
        for params in ({"rank": 0}, {"rank": 1.5}, {"rank": True}, {"iterations": 0},
                       {"alpha": -1}, {"alpha": np.nan}, {"regularization": 0},
                       {"regularization": np.inf}, {"init_std": 0}):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    IALS(**params)


if __name__ == "__main__":
    unittest.main()
