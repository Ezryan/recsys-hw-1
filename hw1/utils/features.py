"""Causal FM features and reproducible negatives using training history only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from utils.evaluation import HistoryContext, build_history_context


USER_FEATURES = ("log1p_review_count", "mean_rating", "positive_fraction", "empty_history")
ITEM_FEATURES = ("log1p_review_count", "mean_rating", "positive_fraction", "log1p_observed_age_days", "empty_history")
FEATURE_NAMES = tuple(f"user.{name}" for name in USER_FEATURES) + tuple(f"item.{name}" for name in ITEM_FEATURES)


class HistoricalFeatureState:
    """State updated only after all queries for a timestamp group have completed."""

    def __init__(self, n_users: int, n_items: int, config: dict[str, Any]):
        if tuple(config["features"]["user"]) != USER_FEATURES or tuple(config["features"]["item"]) != ITEM_FEATURES or config["features"]["include_month"]:
            raise ValueError("FM expects the nine declared historical features without month")
        self.fallback_mean = config["features"]["empty_history_mean_rating"]
        self.fallback_positive = config["features"]["empty_history_positive_fraction"]
        self.threshold = config["evaluation"]["relevance_rating_threshold"]
        self.user_counts = np.zeros(n_users, dtype=np.int64)
        self.item_counts = np.zeros(n_items, dtype=np.int64)
        self.user_sums, self.user_positives = np.zeros(n_users), np.zeros(n_users)
        self.item_sums, self.item_positives = np.zeros(n_items), np.zeros(n_items)
        self.item_first_seen = np.full(n_items, np.iinfo(np.int64).max, dtype=np.int64)
        self.latest_timestamp = None

    def _entity_features(self, indices, entity):
        counts = getattr(self, f"{entity}_counts")[indices]
        sums = getattr(self, f"{entity}_sums")[indices]
        positives = getattr(self, f"{entity}_positives")[indices]
        mean = np.full(len(indices), self.fallback_mean, dtype=np.float64)
        fraction = np.full(len(indices), self.fallback_positive, dtype=np.float64)
        np.divide(sums, counts, out=mean, where=counts > 0)
        np.divide(positives, counts, out=fraction, where=counts > 0)
        return np.column_stack((np.log1p(counts), mean, fraction, counts == 0))

    def user_features(self, users):
        return self._entity_features(np.asarray(users), "user")

    def item_features(self, items, timestamp_ns: int):
        if self.latest_timestamp is not None and timestamp_ns <= self.latest_timestamp:
            raise ValueError("Features must be queried strictly after every applied event group")
        items = np.asarray(items)
        base = self._entity_features(items, "item")
        age = np.zeros(len(items))
        known = self.item_counts[items] > 0
        age[known] = (timestamp_ns - self.item_first_seen[items[known]]) / 86_400_000_000_000
        if (age < 0).any():
            raise ValueError("Item age cannot be negative")
        return np.column_stack((base[:, :3], np.log1p(age), base[:, 3]))

    def query(self, users, items, timestamp_ns: int):
        return np.column_stack((self.user_features(users), self.item_features(items, timestamp_ns)))

    def update(self, users, items, ratings, timestamp_ns: int):
        if self.latest_timestamp is not None and timestamp_ns <= self.latest_timestamp:
            raise ValueError("Apply each timestamp group once in increasing order")
        positives = np.asarray(ratings) >= self.threshold
        for entity, indices in (("user", users), ("item", items)):
            np.add.at(getattr(self, f"{entity}_counts"), indices, 1)
            np.add.at(getattr(self, f"{entity}_sums"), indices, ratings)
            np.add.at(getattr(self, f"{entity}_positives"), indices, positives)
        np.minimum.at(self.item_first_seen, items, timestamp_ns)
        self.latest_timestamp = timestamp_ns


@dataclass
class FMExamples:
    """Local arrays; never publish event IDs, timestamps, or sampled pairs."""

    users: np.ndarray
    items: np.ndarray
    labels: np.ndarray
    raw_features: np.ndarray
    timestamps_ns: np.ndarray
    anchor_source_lines: np.ndarray
    sampled: np.ndarray
    snapshot_users: np.ndarray
    snapshot_items: np.ndarray
    report: dict[str, Any]


def build_fm_examples(history: pd.DataFrame, context: HistoryContext, config: dict[str, Any], *, negatives_per_positive: int) -> FMExamples:
    """Uniform distinct negatives per anchor from the strictly prior positive catalog.

    Exclude all prior seen items (including low ratings) and the anchor itself.
    Other simultaneous/future interactions are deliberately not consulted.
    The same seed/q/history produces exactly the same examples for FM controls.
    """
    if isinstance(negatives_per_positive, (bool, np.bool_)) or not isinstance(negatives_per_positive, (int, np.integer)) or negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be a nonnegative integer")
    if build_history_context(history, context.cutoff, config).fingerprint != context.fingerprint:
        raise ValueError("FM history does not match its context")
    frame = history.sort_values(["timestamp", "source_line"]).reset_index(drop=True)
    users = pd.Index(context.user_ids).get_indexer(frame["user_id"])
    items = pd.Index(context.item_ids).get_indexer(frame["item_id"])
    ratings = frame["rating"].to_numpy(dtype=np.float64)
    times = frame["timestamp"].astype("int64").to_numpy()
    lines = frame["source_line"].to_numpy(dtype=np.int64)
    state = HistoricalFeatureState(len(context.user_ids), len(context.item_ids), config)
    rng = np.random.default_rng(config["seed"])
    seen = [set() for _ in context.user_ids]
    active, active_set = [], set()
    chunks = {name: [] for name in ("users", "items", "labels", "raw_features", "timestamps_ns", "anchor_source_lines", "sampled")}
    requested, generated, shortage_anchors, no_catalog_anchors = 0, 0, 0, 0
    boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1, len(frame)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        timestamp = int(times[left])
        group_u, group_i, group_r = users[left:right], items[left:right], ratings[left:right]
        positive = group_r >= state.threshold
        negative_u, negative_i, negative_lines = [], [], []
        for position in np.flatnonzero(positive):
            user, anchor = int(group_u[position]), int(group_i[position])
            forbidden = seen[user] | {anchor}
            available = len(active) - sum(item in active_set for item in forbidden)
            take = min(negatives_per_positive, available)
            requested += negatives_per_positive
            shortage_anchors += int(take < negatives_per_positive)
            no_catalog_anchors += int(negatives_per_positive > 0 and len(active) == 0)
            if take == 0:
                continue
            if available < len(active) / 4 or take > available / 2:
                eligible = np.array([item for item in active if item not in forbidden], dtype=np.int64)
                selected = rng.choice(eligible, size=take, replace=False).tolist()
            else:
                selected, selected_set = [], set()
                while len(selected) < take:
                    item = active[int(rng.integers(len(active)))]
                    if item not in forbidden and item not in selected_set:
                        selected.append(item)
                        selected_set.add(item)
            negative_u.extend([user] * take)
            negative_i.extend(selected)
            negative_lines.extend([lines[left + position]] * take)
        n_negative = len(negative_i)
        batch_u = np.r_[group_u, np.array(negative_u, dtype=np.int64)]
        batch_i = np.r_[group_i, np.array(negative_i, dtype=np.int64)]
        values = {
            "users": batch_u, "items": batch_i,
            "labels": np.r_[positive.astype(np.float32), np.zeros(n_negative, dtype=np.float32)],
            "raw_features": state.query(batch_u, batch_i, timestamp).astype(np.float32),
            "timestamps_ns": np.full(len(batch_u), timestamp, dtype=np.int64),
            "anchor_source_lines": np.r_[lines[left:right], np.array(negative_lines, dtype=np.int64)],
            "sampled": np.r_[np.zeros(right-left, dtype=bool), np.ones(n_negative, dtype=bool)],
        }
        for name, value in values.items():
            chunks[name].append(value)
        generated += n_negative
        # Every example above sees the same pre-group state, never another same-time label.
        state.update(group_u, group_i, group_r, timestamp)
        for user, item in zip(group_u, group_i):
            seen[user].add(int(item))
        for item in np.unique(group_i[positive]):
            if int(item) not in active_set:
                active.append(int(item))
                active_set.add(int(item))
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    digest = hashlib.sha256()
    for name, values in arrays.items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    report = {
        "n_observed": len(frame), "n_observed_positive": int((ratings >= state.threshold).sum()),
        "n_observed_negative": int((ratings < state.threshold).sum()),
        "requested_sampled_negatives": requested, "n_sampled_negatives": generated,
        "anchors_with_negative_shortage": shortage_anchors, "anchors_without_prior_catalog": no_catalog_anchors,
        "n_examples": len(arrays["users"]), "negative_sampling_seed": config["seed"],
        "negative_sampling_policy": "uniform_without_replacement_per_anchor_prior_positive_catalog_minus_prior_seen_and_anchor",
        "same_time_policy": "query_entire_timestamp_group_before_any_state_update",
        "future_pairs_consulted": False, "feature_names": list(FEATURE_NAMES),
        "training_context_sha256": context.fingerprint, "examples_sha256": digest.hexdigest(),
    }
    return FMExamples(**arrays,
        snapshot_users=state.user_features(np.arange(len(context.user_ids))).astype(np.float32),
        snapshot_items=state.item_features(np.arange(len(context.item_ids)), context.cutoff.value).astype(np.float32),
        report=report)


class FeatureScaler:
    """Standardize continuous columns using observed training events only.

    Empty-history indicators stay 0/1; constant continuous columns use scale=1.
    Sampling rate cannot change the normalization statistics.
    """

    def fit(self, observed_features):
        x = np.asarray(observed_features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not len(x) or not np.isfinite(x).all():
            raise ValueError("Scaler expects finite observed training features with nine columns")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.
        for index, name in enumerate(FEATURE_NAMES):
            if name.endswith(".empty_history"):
                self.mean_[index], self.scale_[index] = 0., 1.
        return self

    def transform(self, features):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Fit the feature scaler on observed training events first")
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError("Expected finite features with nine columns")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def summary(self):
        return {"fit_on": "observed_training_events_only", "feature_names": list(FEATURE_NAMES),
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
                "binary_indicators_standardized": False}
