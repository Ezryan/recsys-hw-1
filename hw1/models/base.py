"""Common fit/score/recommend/evaluate contract for the five future models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import time
from typing import Any

import numpy as np
import pandas as pd

from utils.evaluation import (
    EvaluationResult, EvaluationStage, HistoryContext, build_history_context,
    evaluate_scorer, recommend_top_k, validate_recommendations,
)


class BaseRecommender(ABC):
    """Own learned parameters; use history-only integer vocabularies for scoring.

    score(users, items) returns the Cartesian user-by-item block. Use encode_users
    and encode_items for original string IDs. FM receives the recommendation
    timestamp and may receive a precomputed feature lookup through features.
    """

    def __init__(self, **params: Any) -> None:
        self.params = deepcopy(params)
        self.context_: HistoryContext | None = None
        self.config_: dict[str, Any] | None = None
        self.is_fitted_ = False
        self.fit_seconds_: float | None = None

    def fit(
        self, history: pd.DataFrame, config: dict[str, Any], *,
        context: HistoryContext | None = None, cutoff: str | pd.Timestamp | None = None,
    ) -> BaseRecommender:
        """Fit history only; context must match a stage, never its held-out labels.

        The default cutoff is T1. Pass the final stage context or explicit T2 for
        refitting on train+validation. A private context protects shared matrices
        against transformations inside a model implementation.
        """
        self.is_fitted_ = False
        self.context_ = None
        self.config_ = None
        self.fit_seconds_ = None
        when = cutoff if cutoff is not None else (
            context.cutoff if context is not None else config["split"]["validation_start_utc"]
        )
        own_context = build_history_context(history, when, config)
        if context is not None and (
            context.fingerprint != own_context.fingerprint
            or context.user_ids != own_context.user_ids or context.item_ids != own_context.item_ids
        ):
            raise ValueError("Training history or vocabulary differs from the evaluation context")
        self.context_ = own_context
        self.config_ = deepcopy(config)
        self.rng_ = np.random.default_rng(config["seed"])
        start = time.perf_counter()
        try:
            self._fit(history.copy(deep=True), self.config_)
        except Exception:
            self.context_ = None
            self.config_ = None
            raise
        self.fit_seconds_ = time.perf_counter() - start
        self.is_fitted_ = True
        return self

    @abstractmethod
    def _fit(self, history: pd.DataFrame, config: dict[str, Any]) -> None:
        """Train and store parameters on self, using self.context_ and self.rng_."""

    @abstractmethod
    def _score(
        self, user_indices: np.ndarray, item_indices: np.ndarray, *,
        timestamp: pd.Timestamp, features: Any = None,
    ) -> np.ndarray:
        """Return finite scores shaped (len(user_indices), len(item_indices))."""

    def _require_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Fit the model successfully before scoring or recommending")

    @staticmethod
    def _indices(values, size: int, name: str, ndim: int | None = 1) -> np.ndarray:
        indices = np.asarray(values)
        if (ndim is not None and indices.ndim != ndim) or indices.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integer history indices")
        if ((indices < 0) | (indices >= size)).any():
            raise ValueError(f"Unknown {name} index")
        return indices

    def score(self, user_indices, item_indices, *, timestamp=None, features=None) -> np.ndarray:
        self._require_fitted()
        users = self._indices(user_indices, len(self.context_.user_ids), "user")
        items = self._indices(item_indices, len(self.context_.item_ids), "item")
        when = self.context_.cutoff if timestamp is None else pd.Timestamp(timestamp)
        if when.tzinfo is None or when.tz_convert("UTC") != self.context_.cutoff:
            raise ValueError("Scoring timestamp must equal the fitted recommendation cutoff")
        scores = np.asarray(self._score(users, items, timestamp=when.tz_convert("UTC"), features=features), dtype=np.float64)
        if scores.shape != (len(users), len(items)) or not np.isfinite(scores).all():
            raise ValueError("Model score must be a finite user-by-item matrix of the requested shape")
        return scores

    def _check_stage(self, stage: EvaluationStage) -> None:
        self._require_fitted()
        if self.context_.fingerprint != stage.context.fingerprint:
            raise ValueError("Model and evaluation stage use different training histories")
        if self.config_["evaluation"] != stage.report["protocol"]:
            raise ValueError("Model and stage evaluation protocols differ")

    def recommend(
        self, user_indices=None, *, stage: EvaluationStage | None = None, k: int | None = None,
        user_batch_size: int | None = None, item_batch_size: int | None = None,
        timestamp=None, features=None,
    ) -> np.ndarray:
        """Return history item indices; with stage, preserve the exact common cohort."""
        self._require_fitted()
        k = self.config_["evaluation"]["k"] if k is None else k
        if stage is not None:
            self._check_stage(stage)
            if k != stage.report["protocol"]["k"]:
                raise ValueError("k must match the common evaluation protocol")
            if user_indices is not None and not np.array_equal(user_indices, stage.user_indices):
                raise ValueError("Evaluation requires the complete common user order")
            user_indices = stage.user_indices
        elif user_indices is None:
            user_indices = np.flatnonzero(self.context_.warm_user_mask & (self.context_.candidate_counts >= k))
        result = recommend_top_k(
            lambda users, items: self.score(users, items, timestamp=timestamp, features=features),
            self.context_, user_indices, k=k,
            user_batch_size=self.config_["runtime"]["user_batch_size"] if user_batch_size is None else user_batch_size,
            item_batch_size=self.config_["runtime"]["item_batch_size"] if item_batch_size is None else item_batch_size,
        )
        if stage is not None:
            validate_recommendations(result, stage)
        return result

    def evaluate(self, stage: EvaluationStage, *, features=None) -> EvaluationResult:
        self._check_stage(stage)
        return evaluate_scorer(
            lambda users, items: self.score(users, items, features=features), stage, self.config_
        )

    def _encode(self, original_ids, vocabulary: tuple[str, ...], name: str) -> np.ndarray:
        values = np.asarray(original_ids, dtype=object)
        if values.ndim != 1:
            raise ValueError(f"Original {name} IDs must be a vector")
        indices = pd.Index(vocabulary).get_indexer(values)
        if (indices < 0).any():
            raise ValueError(f"Unknown original {name} ID")
        return indices

    def encode_users(self, original_ids) -> np.ndarray:
        self._require_fitted()
        return self._encode(original_ids, self.context_.user_ids, "user")

    def encode_items(self, original_ids) -> np.ndarray:
        self._require_fitted()
        return self._encode(original_ids, self.context_.item_ids, "item")

    def decode_items(self, item_indices) -> np.ndarray:
        self._require_fitted()
        indices = self._indices(item_indices, len(self.context_.item_ids), "item", ndim=None)
        return np.asarray(self.context_.item_ids, dtype=object)[indices]
