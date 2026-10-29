"""Common model contract, isolated training state, and shared ranking/evaluation."""

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from models.base import BaseRecommender
from utils.evaluation import build_evaluation_stage
from utils.reproducibility import load_config


class CountProbe(BaseRecommender):
    """Synthetic interface fixture, not an experimental recommendation model."""

    def _fit(self, history, config):
        if self.params.get("fail"):
            raise RuntimeError("Synthetic fit failure")
        if self.params.get("transform_copy"):
            self.context_.positive.data *= 3
            history.loc[history.index[0], "rating"] = 1.
            config["seed"] = 123
        self.weights_ = np.asarray(self.context_.positive.sum(axis=0)).ravel()

    def _score(self, user_indices, item_indices, *, timestamp, features=None):
        self.last_timestamp_ = timestamp
        if self.params.get("bad_output") == "shape":
            return np.zeros((len(user_indices), len(item_indices) + 1))
        if self.params.get("bad_output") == "nan":
            return np.full((len(user_indices), len(item_indices)), np.nan)
        scores = np.broadcast_to(self.weights_[item_indices], (len(user_indices), len(item_indices))).copy()
        return scores + (features["offset"] if features is not None else 0)


class BaseModelTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["split"].update({"validation_start_utc": "2020-01-02T00:00:00+00:00", "test_start_utc": "2020-01-03T00:00:00+00:00"})
        rows = [("A", "item-00", 5., 1), ("A", "item-01", 1., 2)]
        rows += [("seed", f"item-{i:02d}", 5., i + 3) for i in range(13)]
        columns = ["user_id", "item_id", "rating", "source_line"]
        self.history = pd.DataFrame(rows, columns=columns).assign(timestamp=pd.Timestamp("2020-01-01", tz="UTC"))
        self.future = pd.DataFrame([("A", "item-02", 5., 100)], columns=columns).assign(timestamp=pd.Timestamp("2020-01-02", tz="UTC"))
        self.stage = build_evaluation_stage({"train": self.history, "validation": self.future}, self.config)

    def test_abstract_and_unfitted_models_do_not_score(self):
        with self.assertRaises(TypeError):
            BaseRecommender()
        model = CountProbe()
        self.assertIsNone(model.fit_seconds_)
        self.assertFalse(hasattr(model, "weights_"))
        for call in (lambda: model.score([0], [0]), lambda: model.recommend(),
                     lambda: model.evaluate(self.stage), lambda: model.encode_users(["A"])):
            with self.assertRaises(RuntimeError):
                call()

    def test_fit_isolates_config_history_and_shared_context(self):
        original = self.history.copy(deep=True)
        original_config = deepcopy(self.config)
        original_matrix = self.stage.context.positive.copy()
        model = CountProbe(transform_copy=True)
        self.assertIs(model.fit(self.history, self.config, context=self.stage.context), model)
        self.assertTrue(model.is_fitted_)
        self.assertGreaterEqual(model.fit_seconds_, 0)
        self.assertIsNot(model.context_, self.stage.context)
        self.assertEqual((self.stage.context.positive != original_matrix).nnz, 0)
        pd.testing.assert_frame_equal(self.history, original)
        self.assertEqual(self.config, original_config)

    def test_scores_have_cartesian_shape_and_explicit_metadata_time(self):
        model = CountProbe().fit(self.history, self.config)
        users, items = model.encode_users(["A", "seed"]), model.encode_items(["item-02", "item-03", "item-04"])
        base = model.score(users, items)
        shifted = model.score(users, items, timestamp="2020-01-02T03:00:00+03:00", features={"offset": 2})
        self.assertEqual(base.shape, (2, 3))
        np.testing.assert_allclose(shifted, base + 2)
        self.assertEqual(str(model.last_timestamp_.tz), "UTC")
        for when in ("2020-01-01T00:00:00+00:00", "2020-01-03T00:00:00+00:00", "2020-01-02"):
            with self.assertRaises(ValueError):
                model.score(users, items, timestamp=when)

    def test_original_ids_and_unknown_indices_are_handled_explicitly(self):
        model = CountProbe().fit(self.history, self.config)
        items = model.encode_items(["item-02", "item-03"])
        self.assertEqual(model.decode_items(items[None, :]).tolist(), [["item-02", "item-03"]])
        with self.assertRaisesRegex(ValueError, "Unknown original"):
            model.encode_users(["unknown"])
        for invalid in ([-1], [1000], [0.5], ["A"]):
            with self.assertRaises(ValueError):
                model.score(invalid, [0])

    def test_recommend_and_evaluate_use_the_common_stage(self):
        model = CountProbe().fit(self.history, self.config, context=self.stage.context)
        recommendations = model.recommend(stage=self.stage, user_batch_size=1, item_batch_size=2)
        self.assertEqual(model.decode_items(recommendations).tolist(), [[f"item-{i:02d}" for i in range(2, 12)]])
        result = model.evaluate(self.stage)
        self.assertEqual(result.summary["stage_sha256"], self.stage.report["stage_sha256"])
        self.assertEqual(result.summary["ndcg@10"], 1.)
        self.assertEqual(result.summary["recall@10"], 1.)
        with self.assertRaises(ValueError):
            model.recommend(stage=self.stage, k=9)
        with self.assertRaises(ValueError):
            model.recommend([1], stage=self.stage)

    def test_different_training_history_is_rejected_even_with_same_ids(self):
        different = self.history.copy()
        different.loc[0, "rating"] = 4.
        model = CountProbe()
        with self.assertRaisesRegex(ValueError, "differs"):
            model.fit(different, self.config, context=self.stage.context)
        model.fit(different, self.config)
        with self.assertRaisesRegex(ValueError, "different training histories"):
            model.evaluate(self.stage)

    def test_failed_refit_invalidates_previous_fit(self):
        model = CountProbe().fit(self.history, self.config)
        model.params["fail"] = True
        with self.assertRaisesRegex(RuntimeError, "Synthetic"):
            model.fit(self.history, self.config)
        self.assertFalse(model.is_fitted_)
        self.assertIsNone(model.context_)
        with self.assertRaises(RuntimeError):
            model.score([0], [0])

    def test_invalid_model_outputs_fail_before_recommendation(self):
        for output in ("shape", "nan"):
            with self.subTest(output=output):
                model = CountProbe(bad_output=output).fit(self.history, self.config)
                with self.assertRaisesRegex(ValueError, "finite user-by-item"):
                    model.score([0], [0, 1])


if __name__ == "__main__":
    unittest.main()
