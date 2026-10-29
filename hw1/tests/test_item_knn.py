"""Analytical and dense-reference checks for independently implemented item-kNN."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from models import ItemKNN
from utils.evaluation import build_evaluation_stage
from utils.reproducibility import load_config


def make_history(rows):
    return pd.DataFrame([
        {"user_id": user, "item_id": item, "rating": float(rating), "source_line": index + 1,
         "timestamp": pd.Timestamp("2020-01-01", tz="UTC")}
        for index, (user, item, rating) in enumerate(rows)
    ])


class ItemKnnTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(load_config())
        self.config["split"].update({"validation_start_utc": "2020-01-02T00:00:00+00:00", "test_start_utc": "2020-01-03T00:00:00+00:00"})
        self.rows = [("u0", "a", 5), ("u0", "b", 5), ("u1", "a", 5), ("u1", "b", 5),
                     ("u2", "a", 5), ("u2", "c", 5), ("u3", "c", 5), ("u4", "d", 1)]
        self.history = make_history(self.rows)

    def test_cosine_and_shrinkage_match_hand_calculation(self):
        model = ItemKNN(k_neighbors=10, shrinkage=10, similarity_block_size=2).fit(self.history, self.config)
        expected = np.zeros((4, 4))
        expected[0, 1] = expected[1, 0] = 2 / math.sqrt(6) * 2 / 12
        expected[0, 2] = expected[2, 0] = 1 / math.sqrt(6) / 11
        np.testing.assert_allclose(model.item_norms_, [math.sqrt(3), math.sqrt(2), math.sqrt(2), 0])
        np.testing.assert_allclose(model.similarity_.toarray(), expected, atol=1e-15)
        self.assertEqual(model.training_summary_["zero_norm_items"], 1)

    def test_target_column_top_k_is_asymmetric_and_scores_use_correct_orientation(self):
        model = ItemKNN(k_neighbors=1, shrinkage=0).fit(self.history, self.config)
        expected = np.zeros((4, 4))
        expected[1, 0] = expected[0, 1] = 2 / math.sqrt(6)
        expected[0, 2] = 1 / math.sqrt(6)
        interactions = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        np.testing.assert_allclose(model.similarity_.toarray(), expected)
        np.testing.assert_allclose(model.score(np.arange(5), np.arange(4)), interactions @ expected)
        self.assertEqual(model.score(model.encode_users(["u3"]), model.encode_items(["a"]))[0, 0], 0)
        self.assertEqual(model.training_summary_["max_neighbors_per_target"], 1)
        self.assertGreater((model.similarity_ != model.similarity_.T).nnz, 0)

    def test_neighbor_ties_choose_the_smaller_original_id(self):
        history = make_history([("u0", "a", 5), ("u0", "b", 5), ("u1", "a", 5), ("u1", "c", 5)])
        model = ItemKNN(k_neighbors=1, shrinkage=0).fit(history, self.config)
        self.assertGreater(model.similarity_[1, 0], 0)
        self.assertEqual(model.similarity_[2, 0], 0)

    def test_item_blocks_and_input_permutation_preserve_learned_similarity(self):
        models = [ItemKNN(k_neighbors=2, shrinkage=7, similarity_block_size=block).fit(
            self.history.sample(frac=1, random_state=block), self.config
        ) for block in (1, 2, 100)]
        for model in models[1:]:
            np.testing.assert_array_equal(model.similarity_.toarray(), models[0].similarity_.toarray())
            self.assertEqual(model.training_summary_["similarity_sha256"], models[0].training_summary_["similarity_sha256"])

    def test_sparse_implementation_matches_an_independent_dense_reference(self):
        rng = np.random.default_rng(42)
        interactions = np.zeros((8, 14))
        interactions[:, :13] = rng.random((8, 13)) < .2
        for item in range(13):
            interactions[item % 8, item] = 1
        names = [f"i{i:02d}" for i in range(13)] + ["zz_low"]
        rows = [(f"u{u}", names[i], 5) for u, i in zip(*np.nonzero(interactions))]
        rows.append(("u0", "zz_low", 1))
        model = ItemKNN(k_neighbors=3, shrinkage=7, similarity_block_size=3).fit(make_history(rows), self.config)
        gram = interactions.T @ interactions
        norms = np.sqrt(np.square(interactions).sum(axis=0))
        denominator = np.outer(norms, norms)
        cosine = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
        cosine *= gram / (gram + 7)
        np.fill_diagonal(cosine, 0)
        expected = np.zeros_like(cosine)
        for target in range(14):
            neighbors = sorted((i for i in range(14) if cosine[i, target] > 0), key=lambda i: (-cosine[i, target], i))[:3]
            expected[neighbors, target] = cosine[neighbors, target]
        np.testing.assert_allclose(model.similarity_.toarray(), expected, atol=1e-15)
        users, items = np.array([7, 0, 3]), np.array([5, 2, 13])
        np.testing.assert_allclose(model.score(users, items), (interactions @ expected)[np.ix_(users, items)], atol=1e-15)

    def test_fit_does_not_materialize_a_dense_item_item_matrix(self):
        with patch.object(sparse.csc_matrix, "toarray", side_effect=AssertionError("Dense training matrix")):
            model = ItemKNN(similarity_block_size=1).fit(self.history, self.config)
        self.assertTrue(sparse.isspmatrix_csc(model.similarity_))
        self.assertTrue(np.isfinite(model.similarity_.data).all())
        self.assertTrue((model.similarity_.diagonal() == 0).all())

    def test_known_low_rating_is_masked_even_if_its_raw_score_is_higher(self):
        self.config["evaluation"].update({"k": 1, "min_candidates": 1, "primary_metric": "ndcg@1", "secondary_metrics": ["recall@1"]})
        history = make_history(self.rows + [("u5", "a", 5), ("u5", "b", 1)])
        future = pd.DataFrame([{"user_id": "u5", "item_id": "c", "rating": 5., "source_line": 100,
                                "timestamp": pd.Timestamp("2020-01-02", tz="UTC")}])
        stage = build_evaluation_stage({"train": history, "validation": future}, self.config)
        model = ItemKNN(k_neighbors=10, shrinkage=0).fit(history, self.config, context=stage.context)
        scores = model.score(model.encode_users(["u5"]), model.encode_items(["b", "c"]))
        self.assertGreater(scores[0, 0], scores[0, 1])
        self.assertEqual(model.decode_items(model.recommend(stage=stage)).tolist(), [["c"]])
        self.assertEqual(model.evaluate(stage).summary["ndcg@1"], 1.)

    def test_zero_positive_histories_and_columns_produce_finite_zero_scores(self):
        model = ItemKNN().fit(self.history, self.config)
        np.testing.assert_array_equal(model.score(model.encode_users(["u4"]), np.arange(4)), np.zeros((1, 4)))
        empty = ItemKNN().fit(make_history([("u", "d", 1)]), self.config)
        self.assertEqual(empty.similarity_.nnz, 0)
        self.assertEqual(empty.score([0], [0])[0, 0], 0)

    def test_invalid_hyperparameters_fail_explicitly(self):
        for values in ({"k_neighbors": 0}, {"k_neighbors": 1.5}, {"k_neighbors": True},
                       {"shrinkage": -1}, {"shrinkage": np.nan}, {"shrinkage": np.inf},
                       {"similarity_block_size": 0}, {"similarity_block_size": 1.5}):
            with self.subTest(params=values):
                with self.assertRaises(ValueError):
                    ItemKNN(**values)


if __name__ == "__main__":
    unittest.main()
