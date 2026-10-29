"""Shared history matrices, warm-start cohorts, candidates, and batched ranking.

Models score the deterministic history indices exposed here and share one
evaluation stage. Binary NDCG/Recall are macro-averaged over that common cohort.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse

from utils.cleaning import clean_reviews
from utils.data import obtain_reviews
from utils.reproducibility import PROJECT_ROOT, load_config, resolve_data_dir
from utils.splitting import split_content_sha256, temporal_split


@dataclass
class HistoryContext:
    """Vocabulary and matrices use history only; the catalog is a subset of items."""

    user_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    positive: sparse.csr_matrix
    seen: sparse.csr_matrix
    warm_user_mask: np.ndarray
    catalog_mask: np.ndarray
    catalog_items: np.ndarray
    candidate_counts: np.ndarray
    cutoff: pd.Timestamp
    fingerprint: str


@dataclass
class EvaluationStage:
    period: str
    context: HistoryContext
    user_indices: np.ndarray
    ground_truth: sparse.csr_matrix
    report: dict[str, Any]


@dataclass
class EvaluationResult:
    """Only summary is publishable; per-user rows belong in local artifacts."""

    summary: dict[str, Any]
    per_user: pd.DataFrame


def _check_frame(frame: pd.DataFrame) -> None:
    needed = {"user_id", "item_id", "rating", "timestamp", "source_line"}
    if needed - set(frame.columns):
        raise ValueError("Expected cleaned interaction columns")
    if not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype) or frame["timestamp"].isna().any():
        raise ValueError("Expected non-null timezone-aware timestamps")
    if not frame["source_line"].is_unique:
        raise ValueError("Source lines must be unique")
    for column in ("user_id", "item_id"):
        if frame[column].isna().any() or not frame[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError("IDs must be non-null original strings")


def _binary_matrix(rows, columns, shape, dtype=bool):
    matrix = sparse.coo_matrix((np.ones(len(rows), dtype=dtype), (rows, columns)), shape=shape).tocsr()
    matrix.sum_duplicates()
    matrix.data.fill(1)
    matrix.sort_indices()
    return matrix


def build_history_context(history: pd.DataFrame, cutoff: str | pd.Timestamp, config: dict[str, Any]) -> HistoryContext:
    """Keep all observed IDs, while defining warm users/catalog by positives."""
    _check_frame(history)
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None or history.empty or not history["timestamp"].lt(cutoff).all():
        raise ValueError("Training history must be nonempty and strictly before an aware cutoff")
    if history.duplicated(["user_id", "item_id"]).any():
        raise ValueError("History must contain unique cleaned user-item pairs")
    user_ids = tuple(sorted(history["user_id"].unique()))
    item_ids = tuple(sorted(history["item_id"].unique()))
    rows = pd.Index(user_ids).get_indexer(history["user_id"])
    columns = pd.Index(item_ids).get_indexer(history["item_id"])
    shape = (len(user_ids), len(item_ids))
    policy = config["evaluation"]
    if policy["min_history_positive_interactions"] < 1:
        raise ValueError("Warm users must have at least one positive history event")
    positives = history["rating"].ge(policy["relevance_rating_threshold"]).to_numpy()
    positive = _binary_matrix(rows[positives], columns[positives], shape, dtype=np.float32)
    seen = _binary_matrix(rows, columns, shape)
    warm_user_mask = np.asarray(positive.getnnz(axis=1) >= policy["min_history_positive_interactions"])
    catalog_mask = np.asarray(positive.getnnz(axis=0) > 0)
    catalog_items = np.flatnonzero(catalog_mask)
    candidate_counts = len(catalog_items) - seen[:, catalog_items].getnnz(axis=1)
    fingerprint = hashlib.sha256(json.dumps({
        "history_content_sha256": split_content_sha256(history),
        "cutoff_utc": cutoff.tz_convert("UTC").isoformat(),
        "relevance_rating_threshold": policy["relevance_rating_threshold"],
        "min_history_positive_interactions": policy["min_history_positive_interactions"],
    }, sort_keys=True).encode()).hexdigest()
    return HistoryContext(user_ids, item_ids, positive, seen, warm_user_mask, catalog_mask,
                          catalog_items, candidate_counts, cutoff.tz_convert("UTC"), fingerprint)


def build_evaluation_stage(
    periods: dict[str, pd.DataFrame], config: dict[str, Any], period: str = "validation"
) -> EvaluationStage:
    """Build one common cohort. Real test is called only after settings are fixed.

    The test path uses train+validation history and is covered by synthetic tests;
    preparing validation does not read the test frame or its labels.
    """
    if period not in {"validation", "test"}:
        raise ValueError("Evaluation period must be validation or test")
    policy = config["evaluation"]
    if policy["candidate_policy"] != "all_history_positive_items_minus_all_seen" or policy["min_relevant_candidates"] != 1:
        raise ValueError("Unsupported candidate or relevant-user policy")
    if type(policy["k"]) is not int or policy["k"] < 1:
        raise ValueError("k must be a positive integer")
    if policy["negative_candidate_sampling"] or policy["update_history_during_evaluation_period"]:
        raise ValueError("The protocol requires full candidates and a frozen history")
    if policy["cold_start_policy"] != "exclude_and_report" or policy["tie_break"] != "ascending_original_item_id":
        raise ValueError("Unsupported cold-start or tie-breaking policy")
    if policy["min_candidates"] < policy["k"]:
        raise ValueError("min_candidates must be at least k")
    if period == "test" and not config["split"]["refit_on_train_validation"]:
        raise ValueError("The final test protocol requires train+validation history")
    history = periods["train"] if period == "validation" else pd.concat([periods["train"], periods["validation"]])
    cutoff = config["split"]["validation_start_utc" if period == "validation" else "test_start_utc"]
    context = build_history_context(history, cutoff, config)
    held_out = periods[period]
    _check_frame(held_out)
    if not held_out["timestamp"].ge(context.cutoff).all():
        raise ValueError("Held-out events must be at or after the recommendation cutoff")
    if period == "validation" and not held_out["timestamp"].lt(pd.Timestamp(config["split"]["test_start_utc"])).all():
        raise ValueError("Validation events must precede the test cutoff")
    if set(history["source_line"]) & set(held_out["source_line"]):
        raise ValueError("History and held-out events must be disjoint")

    users = pd.Index(context.user_ids).get_indexer(held_out["user_id"])
    items = pd.Index(context.item_ids).get_indexer(held_out["item_id"])
    mapped_users, mapped_items = users >= 0, items >= 0
    warm_users, catalog_items = np.zeros(len(held_out), dtype=bool), np.zeros(len(held_out), dtype=bool)
    warm_users[mapped_users] = context.warm_user_mask[users[mapped_users]]
    catalog_items[mapped_items] = context.catalog_mask[items[mapped_items]]
    known_pair = np.zeros(len(held_out), dtype=bool)
    mapped_pairs = mapped_users & mapped_items
    known_pair[mapped_pairs] = np.asarray(context.seen[users[mapped_pairs], items[mapped_pairs]]).ravel()
    enough_candidates = np.zeros(len(held_out), dtype=bool)
    enough_candidates[mapped_users] = context.candidate_counts[users[mapped_users]] >= policy["min_candidates"]
    positive_events = held_out["rating"].ge(policy["relevance_rating_threshold"]).to_numpy()
    retained = positive_events.copy()
    audit = []
    for name, keep in (("warm_user", warm_users), ("item_in_catalog", catalog_items),
                       ("exclude_all_seen", ~known_pair), ("enough_candidates", enough_candidates)):
        before = int(retained.sum())
        retained &= keep
        audit.append({"rule": name, "input_positive_events": before,
                      "removed_positive_events": before - int(retained.sum()),
                      "remaining_positive_events": int(retained.sum())})
    ground_truth = _binary_matrix(users[retained], items[retained], context.positive.shape)
    evaluated_users = np.unique(users[retained])
    heldout_warm_users = np.unique(users[warm_users])
    heldout_warm_enough_users = np.unique(users[warm_users & enough_candidates])
    all_users, all_items = held_out["user_id"].nunique(), held_out["item_id"].nunique()
    cold_user_count = int(held_out.loc[~warm_users, "user_id"].nunique())
    cold_item_count = int(held_out.loc[~catalog_items, "item_id"].nunique())
    cold_positive = positive_events & (~warm_users | ~catalog_items)
    user_audit = [
        {"rule": "held_out_users", "remaining_users": int(all_users)},
        {"rule": "warm_user", "remaining_users": len(heldout_warm_users)},
        {"rule": "enough_candidates", "remaining_users": len(heldout_warm_enough_users)},
        {"rule": "has_relevant_candidate", "remaining_users": len(evaluated_users)},
    ]
    # Hash the actual common cohort and sparse ground truth, without publishing IDs.
    digest = hashlib.sha256(context.fingerprint.encode())
    for array in (evaluated_users, ground_truth.indptr, ground_truth.indices):
        digest.update(np.asarray(array, dtype="<i8").tobytes())
    digest.update(json.dumps(policy, sort_keys=True).encode())
    report = {
        "period": period, "cutoff_utc": context.cutoff.isoformat(),
        "history_context_sha256": context.fingerprint, "stage_sha256": digest.hexdigest(),
        "protocol": dict(policy), "history_n_users": len(context.user_ids), "history_n_items": len(context.item_ids),
        "history_positive_pairs": context.positive.nnz, "history_seen_pairs": context.seen.nnz,
        "n_warm_history_users": int(context.warm_user_mask.sum()), "n_catalog_items": len(context.catalog_items),
        "held_out_events": len(held_out), "held_out_positive_events": int(positive_events.sum()),
        "n_evaluation_users": len(evaluated_users), "n_relevant_pairs": ground_truth.nnz,
        "mean_candidates_per_evaluation_user": float(context.candidate_counts[evaluated_users].mean()) if len(evaluated_users) else None,
        "min_candidates_per_evaluation_user": int(context.candidate_counts[evaluated_users].min()) if len(evaluated_users) else None,
        "cold_start": {
            "cold_user_ids": cold_user_count, "fraction_held_out_user_ids": cold_user_count / all_users if all_users else 0.,
            "new_user_ids": int(held_out.loc[~mapped_users, "user_id"].nunique()),
            "known_users_without_positive_history": int(held_out.loc[mapped_users & ~warm_users, "user_id"].nunique()),
            "out_of_catalog_item_ids": cold_item_count, "fraction_held_out_item_ids": cold_item_count / all_items if all_items else 0.,
            "new_item_ids": int(held_out.loc[~mapped_items, "item_id"].nunique()),
            "known_items_without_positive_history": int(held_out.loc[mapped_items & ~catalog_items, "item_id"].nunique()),
            "positive_events_with_cold_user": int((positive_events & ~warm_users).sum()),
            "positive_events_with_out_of_catalog_item": int((positive_events & ~catalog_items).sum()),
            "positive_events_with_either": int(cold_positive.sum()),
            "fraction_positive_events_with_either": float(cold_positive.sum() / positive_events.sum()) if positive_events.any() else 0.,
        },
        "positive_event_audit": audit, "user_audit": user_audit,
        "n_warm_users_with_too_few_candidates": len(heldout_warm_users) - len(heldout_warm_enough_users),
        "n_warm_users_without_relevant_candidates": len(heldout_warm_enough_users) - len(evaluated_users),
        "test_labels_used": period == "test",
    }
    return EvaluationStage(period, context, evaluated_users, ground_truth, report)


def _stable_top_k(item_indices: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    valid = (item_indices >= 0) & np.isfinite(scores)
    items, values = item_indices[valid], scores[valid]
    if len(items) > k:
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        above = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)
        needed = k - len(above)
        if len(tied) > needed:
            tied = tied[np.argpartition(items[tied], needed - 1)[:needed]]
        selected = np.concatenate([above, tied])
        items, values = items[selected], values[selected]
    order = np.lexsort((items, -values))
    return items[order], values[order]


def recommend_top_k(
    scorer: Callable[[np.ndarray, np.ndarray], np.ndarray], context: HistoryContext,
    user_indices: np.ndarray, k: int = 10, user_batch_size: int = 128, item_batch_size: int = 2048,
) -> np.ndarray:
    """Score full candidates in bounded blocks; ties follow original string IDs.

    scorer(users, items) returns a finite matrix shaped (len(users), len(items)).
    No held-out labels or targets are passed to the scoring/masking mechanism.
    """
    users = np.asarray(user_indices)
    if users.ndim != 1 or users.dtype.kind not in "iu" or len(np.unique(users)) != len(users):
        raise ValueError("User indices must be a unique integer vector")
    if k < 1 or user_batch_size < 1 or item_batch_size < 1:
        raise ValueError("k and batch sizes must be positive")
    if ((users < 0) | (users >= len(context.user_ids))).any():
        raise ValueError("Unknown user index")
    if not context.warm_user_mask[users].all() or not (context.candidate_counts[users] >= k).all():
        raise ValueError("Users must have positive history and at least k candidates")
    output = np.empty((len(users), k), dtype=np.int64)
    for start in range(0, len(users), user_batch_size):
        batch = users[start:start + user_batch_size]
        best_items = np.full((len(batch), k), -1, dtype=np.int64)
        best_scores = np.full((len(batch), k), -np.inf)
        for item_start in range(0, len(context.catalog_items), item_batch_size):
            items = context.catalog_items[item_start:item_start + item_batch_size]
            scores = np.array(scorer(batch, items), dtype=np.float64, copy=True)
            if scores.shape != (len(batch), len(items)) or not np.isfinite(scores).all():
                raise ValueError("Scorer must return a finite user-by-item matrix of the expected shape")
            scores[context.seen[batch][:, items].toarray()] = -np.inf
            merged_items = np.concatenate([best_items, np.broadcast_to(items, scores.shape)], axis=1)
            merged_scores = np.concatenate([best_scores, scores], axis=1)
            best_items.fill(-1)
            best_scores.fill(-np.inf)
            for row in range(len(batch)):
                chosen, values = _stable_top_k(merged_items[row], merged_scores[row], k)
                best_items[row, :len(chosen)] = chosen
                best_scores[row, :len(chosen)] = values
        if (best_items < 0).any():
            raise ValueError("Fewer than k valid candidates survived scoring")
        output[start:start + len(batch)] = best_items
    return output


def validate_recommendations(recommendations: np.ndarray, stage: EvaluationStage) -> dict[str, bool]:
    """Require exactly k unique, unseen catalog items in the common user order."""
    values = np.asarray(recommendations)
    k = stage.report["protocol"]["k"]
    if values.shape != (len(stage.user_indices), k) or values.dtype.kind not in "iu":
        raise ValueError("Recommendations must be an integer matrix with one k-item row per common user")
    checks = {
        "exactly_k_per_common_user": True,
        "unique_items": bool((np.diff(np.sort(values, axis=1), axis=1) != 0).all()),
        "all_items_in_catalog": bool(np.isin(values, stage.context.catalog_items).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Invalid recommendations: {checks}")
    seen = np.asarray(stage.context.seen[np.repeat(stage.user_indices, k), values.ravel()]).ravel()
    checks["all_items_unseen"] = bool(not seen.any())
    if not checks["all_items_unseen"]:
        raise ValueError("Recommendations contain already known items")
    return checks


def evaluate_recommendations(recommendations: np.ndarray, stage: EvaluationStage) -> EvaluationResult:
    """Compute binary NDCG@K and Recall@K on exactly the stage's user cohort.

    Empty/invalid ground truth is an error: users must be filtered by the common
    stage builder, never independently for a model during metric computation.
    Dense intermediates are only n_evaluation_users by K, not user by catalog.
    """
    users = np.asarray(stage.user_indices)
    if len(users) == 0:
        raise ValueError("Cannot average metrics over an empty evaluation cohort")
    if users.ndim != 1 or users.dtype.kind not in "iu" or len(np.unique(users)) != len(users):
        raise ValueError("The common evaluation cohort must contain unique integer indices")
    if ((users < 0) | (users >= len(stage.context.user_ids))).any():
        raise ValueError("The evaluation cohort contains an unknown user")
    k = stage.report["protocol"]["k"]
    validate_recommendations(recommendations, stage)
    truth = stage.ground_truth
    if not sparse.isspmatrix_csr(truth) or truth.shape != stage.context.positive.shape:
        raise ValueError("Ground truth must be CSR with the common history vocabulary")
    if not truth.has_canonical_format or not np.equal(truth.data, 1).all():
        raise ValueError("Ground truth must contain unique binary relevant pairs, without stored zeros")
    relevant_counts = truth.getnnz(axis=1)[users]
    if (relevant_counts <= 0).any():
        raise ValueError("Every common user must have a positive number of relevant candidates")
    nonempty_rows = np.flatnonzero(truth.getnnz(axis=1))
    if not np.array_equal(nonempty_rows, np.sort(users)):
        raise ValueError("Ground truth and the common evaluation cohort disagree")
    truth_rows = np.repeat(np.arange(truth.shape[0]), np.diff(truth.indptr))
    if not stage.context.catalog_mask[truth.indices].all():
        raise ValueError("Ground truth contains items outside the candidate catalog")
    if np.asarray(stage.context.seen[truth_rows, truth.indices]).any():
        raise ValueError("Ground truth contains already known items")

    recommendations = np.asarray(recommendations)
    hits = np.asarray(truth[np.repeat(users, k), recommendations.ravel()]).reshape(len(users), k).astype(np.float64)
    hit_counts = hits.sum(axis=1).astype(np.int64)
    discounts = 1. / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = np.cumsum(hits * discounts, axis=1)[:, -1]
    ideal_prefix = np.concatenate(([0.], np.cumsum(discounts)))
    idcg = ideal_prefix[np.minimum(relevant_counts, k)]
    recall = hit_counts / relevant_counts
    ndcg = dcg / idcg
    per_user = pd.DataFrame({
        "user_index": users, "n_relevant": relevant_counts, f"hits@{k}": hit_counts,
        f"recall@{k}": recall, f"dcg@{k}": dcg, f"idcg@{k}": idcg, f"ndcg@{k}": ndcg,
    })
    summary = {
        "period": stage.period, "k": k, "aggregation": "macro_user", "n_users": len(users),
        "n_relevant_pairs": int(relevant_counts.sum()), "stage_sha256": stage.report["stage_sha256"],
        f"ndcg@{k}": float(ndcg.mean()), f"recall@{k}": float(recall.mean()),
    }
    return EvaluationResult(summary, per_user)


def evaluate_scorer(
    scorer: Callable[[np.ndarray, np.ndarray], np.ndarray], stage: EvaluationStage, config: dict[str, Any]
) -> EvaluationResult:
    """Shared model evaluation: bounded full-catalog ranking, checks, and metrics."""
    if config["evaluation"] != stage.report["protocol"]:
        raise ValueError("Evaluation configuration differs from the common stage protocol")
    recommendations = recommend_top_k(
        scorer, stage.context, stage.user_indices, k=stage.report["protocol"]["k"],
        user_batch_size=config["runtime"]["user_batch_size"], item_batch_size=config["runtime"]["item_batch_size"],
    )
    return evaluate_recommendations(recommendations, stage)


def metric_sanity_checks(config: dict[str, Any]) -> pd.DataFrame:
    """Analytic K=10 examples on synthetic events, not model quality results."""
    synthetic = deepcopy(config)
    synthetic["evaluation"].update({"k": 10, "min_candidates": 10})
    synthetic["split"].update({
        "validation_start_utc": "2020-01-02T00:00:00+00:00",
        "test_start_utc": "2020-01-03T00:00:00+00:00",
    })
    ideal_ten = sum(1 / math.log2(rank + 1) for rank in range(1, 11))
    cases = [
        ("ideal_three_relevant", [set(range(3))], [list(range(10))], 1., 1.),
        ("no_hits", [{20, 21}], [list(range(10))], 0., 0.),
        ("one_hit_at_rank_1", [{0}], [list(range(10))], 1., 1.),
        ("one_hit_at_rank_10", [{0}], [list(range(1, 10)) + [0]], 1 / math.log2(11), 1.),
        ("two_hits_at_ranks_1_and_3", [{0, 2}], [list(range(10))], 1.5 / (1 + 1 / math.log2(3)), 1.),
        ("twenty_relevant", [set(range(20))], [list(range(10))], 1., .5),
        ("one_of_three_at_rank_2", [{0, 20, 21}], [[1, 0] + list(range(2, 10))],
         (1 / math.log2(3)) / (1 + 1 / math.log2(3) + .5), 1/3),
        ("macro_two_unequal_histories", [{0}, {0, *range(10, 29)}], [list(range(10)), list(range(10))],
         (1 + 1 / ideal_ten) / 2, (1 + 1/20) / 2),
    ]
    rows = []
    for name, relevant_sets, predictions, expected_ndcg, expected_recall in cases:
        history_rows = [("synthetic-seed", f"synthetic-item-{i:02d}", 5., i + 1) for i in range(40)]
        history_rows += [(f"synthetic-user-{u}", "synthetic-item-40", 5., 100 + u) for u in range(len(relevant_sets))]
        held_rows = [(f"synthetic-user-{u}", f"synthetic-item-{item:02d}", 5., 1000 + u * 100 + item)
                     for u, relevant in enumerate(relevant_sets) for item in sorted(relevant)]
        columns = ["user_id", "item_id", "rating", "source_line"]
        history = pd.DataFrame(history_rows, columns=columns).assign(timestamp=pd.Timestamp("2020-01-01", tz="UTC"))
        held = pd.DataFrame(held_rows, columns=columns).assign(timestamp=pd.Timestamp("2020-01-02", tz="UTC"))
        stage = build_evaluation_stage({"train": history, "validation": held}, synthetic)
        mapping = {item: index for index, item in enumerate(stage.context.item_ids)}
        recommended = np.array([[mapping[f"synthetic-item-{item:02d}"] for item in row] for row in predictions])
        actual = evaluate_recommendations(recommended, stage).summary
        passed = math.isclose(actual["ndcg@10"], expected_ndcg, rel_tol=0, abs_tol=1e-12) and math.isclose(
            actual["recall@10"], expected_recall, rel_tol=0, abs_tol=1e-12
        )
        if not passed:
            raise ValueError(f"Metric sanity check failed: {name}")
        rows.append({"case": name, "n_users": len(relevant_sets), "expected_ndcg@10": expected_ndcg,
                     "actual_ndcg@10": actual["ndcg@10"], "expected_recall@10": expected_recall,
                     "actual_recall@10": actual["recall@10"], "passed": passed})
    return pd.DataFrame(rows)


def save_evaluation_stage(stage: EvaluationStage, config: dict[str, Any]) -> Path:
    """Save individual matrices/mappings locally and publish an aggregate report."""
    output = resolve_data_dir(config) / "evaluation" / stage.period
    output.mkdir(parents=True, exist_ok=True)
    context = stage.context
    for name, matrix in (("history_positive", context.positive), ("history_seen", context.seen), ("ground_truth", stage.ground_truth)):
        sparse.save_npz(output / f"{name}.npz", matrix)
    (output / "mappings.json").write_text(json.dumps({
        "user_ids": context.user_ids, "item_ids": context.item_ids,
        "catalog_item_indices": context.catalog_items.tolist(),
        "warm_user_indices": np.flatnonzero(context.warm_user_mask).tolist(),
        "evaluation_user_indices": stage.user_indices.tolist(),
        "stage_sha256": stage.report["stage_sha256"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = PROJECT_ROOT / config["storage"]["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    (results / f"evaluation_{stage.period}_summary.json").write_text(
        json.dumps(stage.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--check-metrics", action="store_true", help="Run analytic synthetic cases without loading data")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_metrics:
        checks = metric_sanity_checks(config)
        results = PROJECT_ROOT / config["storage"]["results_dir"]
        results.mkdir(parents=True, exist_ok=True)
        checks.to_csv(results / "metric_sanity_checks.csv", index=False)
        print(checks.to_string(index=False))
        return
    raw, _ = obtain_reviews(config)
    cleaned, _ = clean_reviews(raw, config)
    periods, _ = temporal_split(cleaned, config)
    stage = build_evaluation_stage(periods, config, "validation")
    save_evaluation_stage(stage, config)
    recommendations = recommend_top_k(
        lambda users, items: np.zeros((len(users), len(items))), stage.context, stage.user_indices,
        k=config["evaluation"]["k"], user_batch_size=config["runtime"]["user_batch_size"],
        item_batch_size=config["runtime"]["item_batch_size"],
    )
    print(json.dumps({"n_evaluation_users": len(stage.user_indices),
                      "n_catalog_items": len(stage.context.catalog_items), "n_relevant_pairs": stage.ground_truth.nnz,
                      "recommendation_checks": validate_recommendations(recommendations, stage),
                      "structural_check_only": True, "test_labels_used": False}, indent=2))


if __name__ == "__main__":
    main()
