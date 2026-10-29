"""Constrained ridge references, negative weights, resource guards, and cache integrity."""
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from models import EASE
from models.ease import ease_objective
from utils.evaluation import build_evaluation_stage
from utils.reproducibility import load_config


def history_frame(rows):
    return pd.DataFrame([
        {"user_id": u, "item_id": i, "rating": float(r), "source_line": n + 1,
         "timestamp": pd.Timestamp("2020-01-01", tz="UTC")}
        for n, (u, i, r) in enumerate(rows)
    ])


class EaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = deepcopy(load_config())
        self.config["storage"]["cache_dir"] = self.directory.name
        self.config["split"].update(validation_start_utc="2020-01-02T00:00:00+00:00", test_start_utc="2020-01-03T00:00:00+00:00")
        self.history = history_frame([("u0", "a", 5), ("u0", "b", 4), ("u1", "a", 4), ("u1", "c", 5)])

    def test_hand_solution_divides_each_target_column_by_its_own_diagonal(self):
        history = history_frame([("u0", "a", 5), ("u0", "b", 4), ("u1", "a", 5)])
        model = EASE(regularization=2, solve_block_size=1).fit(history, self.config)
        np.testing.assert_allclose(model.weights_, [[0, .25], [1/3, 0]], atol=1e-14)

    def test_negative_coefficients_are_preserved_and_score_orientation_is_correct(self):
        model = EASE(regularization=1, solve_block_size=2).fit(self.history, self.config)
        expected = np.array([[0, .4, .4], [.5, 0, -.2], [.5, -.2, 0]])
        np.testing.assert_allclose(model.weights_, expected, atol=1e-14)
        np.testing.assert_allclose(model.score([1, 0], [2, 0]), np.array([[1., 0., 1.], [1., 1., 0.]]) @ expected[:, [2, 0]])
        self.assertEqual(model.training_summary_["negative_coefficients"], 2)
        self.assertTrue(model.training_summary_["factor_reused_gram_buffer"])
        self.assertFalse(model.training_summary_["full_inverse_materialized"])

    def test_matches_independent_per_target_constrained_ridge(self):
        rng = np.random.default_rng(42)
        r = (rng.random((9, 7)) < .4).astype(float)
        rows = [(f"u{u}", f"i{i}", 5 if r[u, i] else 1) for u in range(9) for i in range(7)]
        history = history_frame(rows)
        self.config["runtime"]["reuse_model_cache"] = False
        expected = np.zeros((7, 7))
        for j in range(7):
            other = np.arange(7) != j
            x = r[:, other]
            expected[other, j] = np.linalg.solve(x.T @ x + 2.3 * np.eye(6), x.T @ r[:, j])
        for block in (1, 3, 100):
            with self.subTest(block=block):
                with patch("numpy.linalg.inv", side_effect=AssertionError("Full inverse is forbidden")):
                    model = EASE(regularization=2.3, solve_block_size=block).fit(history, self.config)
                np.testing.assert_allclose(model.weights_, expected, atol=1e-14)

    def test_block_objective_equals_direct_dense_loss(self):
        model = EASE(regularization=1).fit(self.history, self.config)
        r = model.interactions_.toarray()
        expected = np.square(r - r @ model.weights_).sum() + np.square(model.weights_).sum()
        for block in (1, 100):
            actual = ease_objective(model.interactions_, model.weights_, regularization=1, block_size=block)
            self.assertAlmostEqual(actual["objective"], expected)
            self.assertAlmostEqual(actual["objective"], model.training_summary_["objective"])
        self.assertLess(expected, np.square(r).sum())

    def test_zero_positive_items_and_empty_positive_histories_have_zero_weights(self):
        history = pd.concat([self.history, history_frame([("u2", "d", 1)])], ignore_index=True)
        history["source_line"] = np.arange(1, len(history) + 1)
        model = EASE(regularization=1).fit(history, self.config)
        np.testing.assert_array_equal(model.weights_[3], np.zeros(4))
        np.testing.assert_array_equal(model.weights_[:, 3], np.zeros(4))
        history["rating"] = 1.
        model = EASE(regularization=1).fit(history, self.config)
        np.testing.assert_array_equal(model.weights_, np.zeros((4, 4)))
        self.assertEqual(model.training_summary_["objective"], 0.)

    def test_memory_guard_runs_before_dense_factorization(self):
        with patch("models.ease.psutil.virtual_memory") as memory, patch("models.ease.linalg.cho_factor") as factor:
            memory.return_value.available = 1024
            with self.assertRaises(MemoryError):
                EASE().fit(self.history, self.config)
            factor.assert_not_called()

    def test_reused_cache_is_memory_mapped_and_skips_training(self):
        first = EASE(regularization=1).fit(self.history, self.config)
        with patch("models.ease.linalg.cho_factor", side_effect=AssertionError("Cache must bypass training")):
            second = EASE(regularization=1).fit(self.history.sample(frac=1, random_state=3), self.config)
        self.assertFalse(first.cache_hit_)
        self.assertTrue(second.cache_hit_)
        self.assertIsInstance(second.weights_, np.memmap)
        np.testing.assert_array_equal(first.weights_, second.weights_)
        self.assertEqual(first.training_compute_seconds_, second.training_compute_seconds_)

    def test_cache_identity_tracks_params_seed_features_and_split(self):
        first = EASE().fit(self.history, self.config)
        directory, key = first._cache_location(self.config)
        for field in ("seed", "features", "split"):
            changed = deepcopy(self.config)
            if field == "seed": changed[field] += 1
            elif field == "features": changed[field]["include_month"] = True
            else: changed[field]["test_start_utc"] = "2020-01-04T00:00:00+00:00"
            self.assertNotEqual(key, first._cache_location(changed)[1])
        second = EASE(regularization=1).fit(self.history, self.config)
        self.assertNotEqual(directory, second.cache_path_)

    def test_corrupted_cache_is_rejected_and_explicit_recompute_repairs_it(self):
        first = EASE().fit(self.history, self.config)
        weights = np.load(first.cache_path_ / "weights.npy", mmap_mode="r+")
        weights[0, 1] += .1
        weights.flush()
        del weights
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            EASE().fit(self.history, self.config)
        self.config["runtime"]["reuse_model_cache"] = False
        repaired = EASE().fit(self.history, self.config)
        self.assertFalse(repaired.cache_hit_)
        np.testing.assert_array_equal(repaired.weights_, first.weights_)

    def test_common_recommendations_exclude_both_positive_and_low_rated_history(self):
        rows = [("A", "item-00", 5), ("A", "item-01", 1)]
        rows += [("seed", f"item-{i:02d}", 5) for i in range(13)]
        history = history_frame(rows)
        future = pd.DataFrame([{"user_id": "A", "item_id": "item-02", "rating": 5., "source_line": 100,
                                "timestamp": pd.Timestamp("2020-01-02", tz="UTC")}])
        stage = build_evaluation_stage({"train": history, "validation": future}, self.config)
        model = EASE().fit(history, self.config, context=stage.context)
        recommended = model.decode_items(model.recommend(stage=stage)).ravel().tolist()
        self.assertEqual(len(set(recommended)), 10)
        self.assertNotIn("item-00", recommended)
        self.assertNotIn("item-01", recommended)
        self.assertEqual(model.evaluate(stage).summary["stage_sha256"], stage.report["stage_sha256"])

    def test_invalid_hyperparameters_are_rejected(self):
        for params in ({"regularization": 0}, {"regularization": -1}, {"regularization": np.inf},
                       {"regularization": np.nan}, {"regularization": True}, {"solve_block_size": 0},
                       {"solve_block_size": 1.5}, {"solve_block_size": True}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                EASE(**params)


if __name__ == "__main__":
    unittest.main()
