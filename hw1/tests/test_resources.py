"""Resource calculations and reproducible train-only fragment selection."""

from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from utils.evaluation import build_history_context
from utils.reproducibility import load_config
from utils.resources import estimate_ease_memory, select_probe_history


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["split"]["validation_start_utc"] = "2020-01-02T00:00:00+00:00"
        self.config["runtime"]["resource_probe"].update({"max_users": 3, "max_items": 4, "ease_solve_block_size": 2})
        self.history = pd.DataFrame([
            (f"user-{u}", f"item-{i}", 5., u * 8 + i + 1) for u in range(5) for i in range(8)
        ], columns=["user_id", "item_id", "rating", "source_line"]).assign(timestamp=pd.Timestamp("2020-01-01", tz="UTC"))

    def test_fragment_is_bounded_and_independent_of_input_order(self):
        first = select_probe_history(self.history, self.config)
        second = select_probe_history(self.history.sample(frac=1, random_state=123), self.config)
        pd.testing.assert_frame_equal(first, second)
        self.assertLessEqual(first["user_id"].nunique(), 3)
        self.assertLessEqual(first["item_id"].nunique(), 4)
        self.assertTrue(set(first["source_line"]) <= set(self.history["source_line"]))

    def test_future_events_cannot_enter_the_resource_probe(self):
        history = self.history.copy()
        history.loc[0, "timestamp"] = pd.Timestamp("2020-01-02", tz="UTC")
        with self.assertRaisesRegex(ValueError, "train history only"):
            select_probe_history(history, self.config)

    def test_dense_memory_formula_and_dynamic_budget(self):
        context = build_history_context(self.history, self.config["split"]["validation_start_utc"], self.config)
        estimate = estimate_ease_memory(context, self.config, available_bytes=8 * 2**30)
        self.assertEqual(estimate["one_float64_matrix_bytes"], 8 * 8 * 8)
        self.assertEqual(estimate["solve_arrays_bytes"], 2 * 8 * 8 * 2)
        self.assertTrue(estimate["fits_optimized_strategy"])
        limited = estimate_ease_memory(context, self.config, available_bytes=1024)
        self.assertFalse(limited["fits_optimized_strategy"])
        self.assertEqual(limited["effective_extra_memory_budget_bytes"], 512)
        self.assertLessEqual(estimate["sparse_gram_nnz_upper_bound"], 8 * 8)


if __name__ == "__main__":
    unittest.main()
