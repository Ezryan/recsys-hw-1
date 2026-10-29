"""Item-based kNN from binary positive history, cosine similarity and shrinkage."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from models.base import BaseRecommender


class ItemKNN(BaseRecommender):
    """S[source, target] keeps the target's strongest positive neighbors.

    Predictions are unnormalized sums R[user, :] @ S. Only the common history
    supplies interactions; known-item masking and top-10 come from the base class.
    """

    def __init__(self, k_neighbors: int = 50, shrinkage: float = 10., similarity_block_size: int = 256) -> None:
        params = self._validated_params(k_neighbors, shrinkage, similarity_block_size)
        super().__init__(**params)

    @staticmethod
    def _validated_params(k_neighbors, shrinkage, similarity_block_size):
        for name, value in (("k_neighbors", k_neighbors), ("similarity_block_size", similarity_block_size)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(shrinkage, (bool, np.bool_)) or not np.isscalar(shrinkage):
            raise ValueError("shrinkage must be finite and nonnegative")
        shrinkage = float(shrinkage)
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        return {"k_neighbors": int(k_neighbors), "shrinkage": shrinkage, "similarity_block_size": int(similarity_block_size)}

    def _fit(self, history: pd.DataFrame, config: dict[str, Any]) -> None:
        self.params = self._validated_params(**self.params)
        k, shrinkage, block = (self.params[key] for key in ("k_neighbors", "shrinkage", "similarity_block_size"))
        self.interactions_ = self.context_.positive.astype(np.float64)
        columns = self.interactions_.tocsc()
        transpose = columns.T.tocsr()
        self.item_norms_ = np.sqrt(np.asarray(self.interactions_.power(2).sum(axis=0)).ravel())
        n_items = columns.shape[1]
        indptr = np.zeros(n_items + 1, dtype=np.int64)
        indices, data = [], []
        max_block_nnz = 0
        for left in range(0, n_items, block):
            right = min(left + block, n_items)
            cooccurrence = (transpose @ columns[:, left:right]).tocsc()
            max_block_nnz = max(max_block_nnz, cooccurrence.nnz)
            for local, target in enumerate(range(left, right)):
                start, end = cooccurrence.indptr[local:local + 2]
                neighbors = cooccurrence.indices[start:end]
                counts = cooccurrence.data[start:end]
                valid = (neighbors != target) & (counts > 0) & (self.item_norms_[neighbors] > 0)
                if self.item_norms_[target] > 0 and valid.any():
                    neighbors, counts = neighbors[valid], counts[valid]
                    weights = counts / (self.item_norms_[neighbors] * self.item_norms_[target])
                    if shrinkage:
                        weights *= counts / (counts + shrinkage)
                    # Indices follow original lexicographic IDs, including ties.
                    selected = np.lexsort((neighbors, -weights))[:k]
                    indices.append(neighbors[selected])
                    data.append(weights[selected])
                    indptr[target + 1] = indptr[target] + len(selected)
                else:
                    indptr[target + 1] = indptr[target]
        self.similarity_ = sparse.csc_matrix((
            np.concatenate(data) if data else np.empty(0, dtype=np.float64),
            np.concatenate(indices) if indices else np.empty(0, dtype=np.int32),
            indptr,
        ), shape=(n_items, n_items))
        self.similarity_.sort_indices()
        digest = hashlib.sha256()
        for array, dtype in ((self.similarity_.data, "<f8"), (self.similarity_.indices, "<i8"), (self.similarity_.indptr, "<i8")):
            digest.update(np.asarray(array, dtype=dtype).tobytes())
        self.training_summary_ = {
            "n_users": self.interactions_.shape[0], "n_history_items": n_items,
            "n_positive_interactions": self.interactions_.nnz,
            "zero_norm_items": int((self.item_norms_ == 0).sum()),
            "similarity_nnz": self.similarity_.nnz,
            "max_neighbors_per_target": int(self.similarity_.getnnz(axis=0).max(initial=0)),
            "similarity_storage_bytes": int(sum(array.nbytes for array in (
                self.similarity_.data, self.similarity_.indices, self.similarity_.indptr
            ))),
            "max_cooccurrence_block_nnz": max_block_nnz,
            "similarity_sha256": digest.hexdigest(),
            "training_context_sha256": self.context_.fingerprint,
            "orientation": "S[source_item, target_item]",
            "score_normalization": "none_sum_of_similarities",
        }

    def _score(self, user_indices, item_indices, *, timestamp, features=None) -> np.ndarray:
        return (self.interactions_[user_indices] @ self.similarity_[:, item_indices]).toarray()
