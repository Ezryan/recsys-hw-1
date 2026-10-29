"""iALS with rating/recency preferences and unchanged binary confidence weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from models.ials import IALS


def validate_target_params(gamma, half_life_days) -> dict[str, Any]:
    if isinstance(gamma, (bool, np.bool_)) or not np.isscalar(gamma):
        raise ValueError("gamma must be a finite nonnegative scalar")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if isinstance(half_life_days, str):
        if half_life_days != "infinity":
            raise ValueError("The only string half-life is 'infinity'")
        half_life_days = np.inf
    if isinstance(half_life_days, (bool, np.bool_)) or not np.isscalar(half_life_days):
        raise ValueError("half_life_days must be positive or infinity")
    half_life_days = float(half_life_days)
    if np.isnan(half_life_days) or half_life_days <= 0:
        raise ValueError("half_life_days must be positive or infinity")
    return {"gamma": gamma, "half_life_days": "infinity" if np.isposinf(half_life_days) else half_life_days}


def custom_preferences(ratings, timestamps, cutoff, *, gamma: float = 1., half_life_days: float | str = 730.) -> np.ndarray:
    """p = ((r-3)/2)^gamma * 2^(-age_days/h) for r>=4; zero otherwise.

    The fixed reference is the history cutoff, never wall-clock time. Numerical
    underflow may produce zero targets; their observed coordinates must be kept.
    """
    params = validate_target_params(gamma, half_life_days)
    ratings = np.asarray(ratings, dtype=np.float64)
    times, cutoff = pd.DatetimeIndex(timestamps), pd.Timestamp(cutoff)
    if ratings.ndim != 1 or len(ratings) != len(times) or not np.isfinite(ratings).all() or ((ratings < 1) | (ratings > 5)).any():
        raise ValueError("Ratings must be finite in [1,5] and match timestamps")
    if times.tz is None or times.hasnans or cutoff.tzinfo is None or pd.isna(cutoff) or not (times < cutoff).all():
        raise ValueError("Event timestamps must be aware and strictly before an aware cutoff")
    ages = np.asarray((cutoff - times).total_seconds(), dtype=np.float64) / 86400
    positive = ratings >= 4
    targets = np.zeros(len(ratings), dtype=np.float64)
    h = np.inf if params["half_life_days"] == "infinity" else params["half_life_days"]
    with np.errstate(over="ignore", under="ignore"):
        targets[positive] = np.power((ratings[positive] - 3) / 2, params["gamma"]) * np.exp2(-ages[positive] / h)
    if not np.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise FloatingPointError("Custom preferences must be finite in [0,1]")
    return targets


def summarize_targets(values) -> dict[str, Any]:
    """Aggregate positive-history target values, including explicit numerical zeros."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Target statistics expect a finite vector")
    summary = {"n_values": len(values), "zero_targets": int(np.count_nonzero(values == 0))}
    statistics = ("min", "q01", "q25", "median", "q75", "q99", "max")
    if len(values):
        summary.update(dict(zip(statistics, np.quantile(values, [0, .01, .25, .5, .75, .99, 1]).tolist())))
        summary["mean"] = float(values.mean())
    else:
        summary.update({name: None for name in (*statistics, "mean")})
    return summary


class CustomIALS(IALS):
    """Change p in the baseline weighted error; keep c=1+alpha on the same Ω+."""

    target_name = "rating_strength_times_cutoff_recency"

    def __init__(self, rank: int = 32, regularization: float = .1, alpha: float = 10.,
                 iterations: int = 10, init_std: float = .01, gamma: float = 1.,
                 half_life_days: float | str = 730.):
        super().__init__(rank=rank, regularization=regularization, alpha=alpha, iterations=iterations, init_std=init_std)
        self.params.update(validate_target_params(gamma, half_life_days))

    def _make_targets(self, history: pd.DataFrame) -> sparse.csr_matrix:
        if self.config_["evaluation"]["relevance_rating_threshold"] != 4:
            raise ValueError("The declared rating-strength formula requires relevance threshold 4")
        preferences = custom_preferences(history["rating"], history["timestamp"], self.context_.cutoff,
                                         gamma=self.params["gamma"], half_life_days=self.params["half_life_days"])
        positive = history["rating"].ge(4).to_numpy()
        rows = pd.Index(self.context_.user_ids).get_indexer(history.loc[positive, "user_id"])
        columns = pd.Index(self.context_.item_ids).get_indexer(history.loc[positive, "item_id"])
        order = np.lexsort((columns, rows))
        expected_rows = np.repeat(np.arange(self.observed_.shape[0]), np.diff(self.observed_.indptr))
        if not np.array_equal(rows[order], expected_rows) or not np.array_equal(columns[order], self.observed_.indices):
            raise ValueError("Custom targets do not align with the baseline positive coordinate pattern")
        targets = self.observed_.copy()
        targets.data[:] = preferences[positive][order]
        # Never eliminate zeros: underflow in p does not remove a confidence weight.
        return targets

    def _fit(self, history: pd.DataFrame, config: dict[str, Any]) -> None:
        self.params.update(validate_target_params(self.params["gamma"], self.params["half_life_days"]))
        super()._fit(history, config)
        self.training_summary_.update({
            "target_formula": "p=((rating-3)/2)^gamma * 2^(-age_days/half_life_days) on positive history; otherwise 0",
            "gamma": self.params["gamma"], "half_life_days": self.params["half_life_days"],
            "target_reference_cutoff_utc": self.context_.cutoff.isoformat(),
            "confidence_changed": False, "observed_coordinates_retained_including_zero_targets": True,
            "target_statistics": summarize_targets(self.targets_.data),
            "recovers_binary_target": self.params["gamma"] == 0 and self.params["half_life_days"] == "infinity",
        })
