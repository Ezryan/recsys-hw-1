"""Implicit ALS with binary preferences and exact weighted least-squares steps."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, sparse

from models.base import BaseRecommender


def _validate_weights(alpha: float, regularization: float) -> None:
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")


def _validate_pattern(observed: sparse.csr_matrix, targets: sparse.csr_matrix) -> None:
    if not sparse.isspmatrix_csr(observed) or not sparse.isspmatrix_csr(targets):
        raise ValueError("Observed interactions and targets must be CSR matrices")
    if not observed.has_canonical_format or not targets.has_canonical_format:
        raise ValueError("CSR inputs must have unique sorted coordinates")
    if (observed.shape != targets.shape or not np.array_equal(observed.indptr, targets.indptr)
            or not np.array_equal(observed.indices, targets.indices)):
        raise ValueError("Targets must retain exactly the observed coordinate pattern, including zero targets")
    if not np.equal(observed.data, 1).all() or not np.isfinite(targets.data).all():
        raise ValueError("Observed entries must be binary ones and target values finite")


def als_update(
    observed: sparse.csr_matrix, targets: sparse.csr_matrix, fixed_factors: np.ndarray,
    *, alpha: float, regularization: float,
) -> np.ndarray:
    """Solve each weighted normal equation with all missing pairs included.

    c=1 everywhere, c=1+alpha on observed coordinates. Targets share that pattern,
    allowing the fifth model to change p without changing the confidence weights.
    """
    _validate_weights(alpha, regularization)
    _validate_pattern(observed, targets)
    fixed = np.asarray(fixed_factors, dtype=np.float64)
    if fixed.ndim != 2 or fixed.shape[0] != observed.shape[1] or fixed.shape[1] < 1 or not np.isfinite(fixed).all():
        raise ValueError("Fixed factors must be a finite item-by-rank matrix matching observed columns")
    rank = fixed.shape[1]
    gram = fixed.T @ fixed
    gram[np.diag_indices(rank)] += regularization
    output = np.zeros((observed.shape[0], rank), dtype=np.float64)
    for row in range(observed.shape[0]):
        start, end = observed.indptr[row:row + 2]
        if start == end:
            continue
        factors = fixed[observed.indices[start:end]]
        system = np.array(gram, order="F", copy=True)
        system += alpha * (factors.T @ factors)
        rhs = factors.T @ ((1 + alpha) * targets.data[start:end])
        cholesky = linalg.cho_factor(system, lower=True, overwrite_a=True, check_finite=False)
        output[row] = linalg.cho_solve(cholesky, rhs, check_finite=False)
    return output


def ials_objective(
    observed: sparse.csr_matrix, targets: sparse.csr_matrix,
    user_factors: np.ndarray, item_factors: np.ndarray,
    *, alpha: float, regularization: float, batch_size: int = 8192,
) -> float:
    """Exact full-pair objective using Gram products plus sparse corrections."""
    _validate_weights(alpha, regularization)
    _validate_pattern(observed, targets)
    x, y = np.asarray(user_factors, dtype=np.float64), np.asarray(item_factors, dtype=np.float64)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != observed.shape[0] or y.shape[0] != observed.shape[1]
            or x.shape[1] != y.shape[1] or x.shape[1] < 1 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Finite factor matrices must match the user/item shape and share a positive rank")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("Objective batch size must be a positive integer")
    loss = float(np.sum((x.T @ x) * (y.T @ y)))
    rows = np.repeat(np.arange(observed.shape[0]), np.diff(observed.indptr))
    for start in range(0, observed.nnz, batch_size):
        end = min(start + batch_size, observed.nnz)
        predictions = np.einsum("ij,ij->i", x[rows[start:end]], y[observed.indices[start:end]])
        loss += float(np.sum((1 + alpha) * np.square(targets.data[start:end] - predictions) - np.square(predictions)))
    loss += regularization * float(np.square(x).sum() + np.square(y).sum())
    return loss


class IALS(BaseRecommender):
    """p=1 for rating>=threshold, otherwise p=0; confidence uses positives only."""

    target_name = "binary_positive_history"

    def __init__(
        self, rank: int = 32, regularization: float = .1, alpha: float = 10.,
        iterations: int = 10, init_std: float = .01,
    ) -> None:
        super().__init__(**self._validated_params(rank, regularization, alpha, iterations, init_std))

    @staticmethod
    def _validated_params(rank, regularization, alpha, iterations, init_std):
        for name, value in (("rank", rank), ("iterations", iterations)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("regularization", regularization), ("alpha", alpha), ("init_std", init_std)):
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite scalar")
        regularization, alpha, init_std = float(regularization), float(alpha), float(init_std)
        _validate_weights(alpha, regularization)
        if not np.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        return {"rank": int(rank), "regularization": regularization, "alpha": alpha,
                "iterations": int(iterations), "init_std": init_std}

    def _make_targets(self, history: pd.DataFrame) -> sparse.csr_matrix:
        """The custom-target model overrides this hook, reusing the same solver."""
        return self.observed_.copy()

    def _fit(self, history: pd.DataFrame, config: dict[str, Any]) -> None:
        names = ("rank", "regularization", "alpha", "iterations", "init_std")
        self.params.update(self._validated_params(**{name: self.params[name] for name in names}))
        rank, regularization, alpha, iterations, init_std = (self.params[name] for name in names)
        self.observed_ = self.context_.positive.astype(np.float64)
        self.targets_ = self._make_targets(history).astype(np.float64).tocsr()
        self.targets_.sort_indices()
        _validate_pattern(self.observed_, self.targets_)
        transpose, target_transpose = self.observed_.T.tocsr(), self.targets_.T.tocsr()
        n_users, n_items = self.observed_.shape
        self.user_factors_ = self.rng_.normal(0, init_std, (n_users, rank))
        self.item_factors_ = self.rng_.normal(0, init_std, (n_items, rank))
        self.loss_history_ = []

        def record(iteration: int, step: str) -> None:
            loss = ials_objective(self.observed_, self.targets_, self.user_factors_, self.item_factors_,
                                  alpha=alpha, regularization=regularization)
            if not np.isfinite(loss):
                raise FloatingPointError("Nonfinite iALS training objective")
            if self.loss_history_ and loss > self.loss_history_[-1]["objective"] + 1e-8 * max(1., abs(self.loss_history_[-1]["objective"])):
                raise FloatingPointError("An ALS update increased the fixed training objective")
            self.loss_history_.append({"iteration": iteration, "step": step, "objective": loss})

        record(0, "initial")
        for iteration in range(1, iterations + 1):
            self.user_factors_ = als_update(self.observed_, self.targets_, self.item_factors_,
                                           alpha=alpha, regularization=regularization)
            record(iteration, "users")
            self.item_factors_ = als_update(transpose, target_transpose, self.user_factors_,
                                           alpha=alpha, regularization=regularization)
            record(iteration, "items")
        digest = hashlib.sha256()
        for values in (self.user_factors_, self.item_factors_):
            digest.update(np.asarray(values, dtype="<f8").tobytes())
        observed_digest, target_digest = hashlib.sha256(), hashlib.sha256()
        for digest_, matrix in ((observed_digest, self.observed_), (target_digest, self.targets_)):
            for values, dtype in ((matrix.indptr, "<i8"), (matrix.indices, "<i8"), (matrix.data, "<f8")):
                digest_.update(np.asarray(values, dtype=dtype).tobytes())
        self.training_summary_ = {
            "n_users": n_users, "n_history_items": n_items, "n_positive_interactions": self.observed_.nnz,
            "zero_positive_users": int((self.observed_.getnnz(axis=1) == 0).sum()),
            "zero_positive_items": int((self.observed_.getnnz(axis=0) == 0).sum()),
            "target_name": self.target_name, "observed_confidence": 1 + alpha, "other_confidence": 1.,
            "initial_objective": self.loss_history_[0]["objective"], "final_objective": self.loss_history_[-1]["objective"],
            "objective_history": self.loss_history_, "nonincreasing_objective": True,
            "factor_storage_bytes": self.user_factors_.nbytes + self.item_factors_.nbytes,
            "factors_sha256": digest.hexdigest(), "observed_pattern_sha256": observed_digest.hexdigest(),
            "target_sha256": target_digest.hexdigest(), "training_context_sha256": self.context_.fingerprint,
            "solver": "own_ALS_normal_equations_with_Cholesky_and_full_Gram",
        }

    def _score(self, user_indices, item_indices, *, timestamp, features=None) -> np.ndarray:
        return self.user_factors_[user_indices] @ self.item_factors_[item_indices].T
