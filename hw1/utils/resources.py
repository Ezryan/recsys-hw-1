"""Isolated CPU kernel probes and conservative full-catalog memory estimates.

Probes use a small train fragment. They are not complete model implementations,
hyperparameter search, or evaluation of recommendation quality.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy import linalg, sparse

from utils.cleaning import CORE_COLUMNS, clean_reviews
from utils.data import obtain_reviews
from utils.evaluation import HistoryContext, build_history_context
from utils.reproducibility import PROJECT_ROOT, configure_reproducibility, load_config, resolve_data_dir
from utils.splitting import temporal_split


def select_probe_history(train: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Seeded user sample among top train-positive items, strictly before T1."""
    cutoff = pd.Timestamp(config["split"]["validation_start_utc"])
    if train.empty or not train["timestamp"].lt(cutoff).all():
        raise ValueError("Resource probe accepts train history only")
    settings = config["runtime"]["resource_probe"]
    positive = train.loc[train["rating"].ge(config["evaluation"]["relevance_rating_threshold"])]
    counts = positive.groupby("item_id").size().rename("count").reset_index()
    items = counts.sort_values(["count", "item_id"], ascending=[False, True]).head(settings["max_items"])["item_id"]
    pool = np.array(sorted(positive.loc[positive["item_id"].isin(items), "user_id"].unique()), dtype=object)
    if len(pool) == 0:
        raise ValueError("Resource probe requires positive train interactions")
    users = np.random.default_rng(config["seed"]).choice(pool, min(settings["max_users"], len(pool)), replace=False)
    return train.loc[train["user_id"].isin(users) & train["item_id"].isin(items), CORE_COLUMNS].sort_values(
        ["timestamp", "source_line"], kind="stable"
    ).reset_index(drop=True)


def estimate_ease_memory(context: HistoryContext, config: dict[str, Any], available_bytes: int) -> dict[str, Any]:
    """Budget extra arrays for in-place Cholesky, B, and block solves.

    Sparse Gram nnz is bounded by the sum of squared user degrees. The bound
    includes conversion overhead, and full inverse/identity matrices are avoided.
    Existing process memory is already excluded from system available memory.
    """
    settings = config["runtime"]["resource_probe"]
    m = len(context.item_ids)  # All history IDs, including zero-positive columns.
    dense = 8 * m * m
    degrees = context.positive.getnnz(axis=1).astype(np.int64)
    gram_nnz_bound = min(m * m, int(np.square(degrees).sum()))
    gram_csr = 12 * gram_nnz_bound + 4 * (m + 1)
    block = min(settings["ease_solve_block_size"], m)
    solve_arrays = 2 * 8 * m * block
    input_csr = 12 * context.positive.nnz + 4 * (len(context.user_ids) + 1)
    workspace = settings["ease_workspace_reserve_mib"] * 2**20
    optimized = max(dense + gram_csr, 2 * dense + solve_arrays) + input_csr + workspace
    budget = min(config["runtime"]["max_dense_memory_mib"] * 2**20,
                 int(available_bytes * config["runtime"]["max_available_memory_fraction"]))
    return {
        "n_history_items": m, "one_float64_matrix_bytes": dense,
        "naive_three_dense_matrices_bytes": 3 * dense, "sparse_gram_nnz_upper_bound": gram_nnz_bound,
        "sparse_gram_bytes_upper_bound": gram_csr, "solve_block_size": block,
        "solve_arrays_bytes": solve_arrays, "workspace_reserve_bytes": workspace,
        "estimated_extra_peak_bytes": optimized, "available_bytes_at_check": available_bytes,
        "effective_extra_memory_budget_bytes": budget, "fits_optimized_strategy": optimized <= budget,
        "strategy": "float64_fortran_gram_inplace_cholesky_plus_B_and_block_solves",
    }


def _prepare_kernel(name: str, history: pd.DataFrame, context: HistoryContext, config: dict[str, Any]):
    settings = config["runtime"]["resource_probe"]
    rank, alpha, regularization = settings["rank"], 10., .1
    rng = np.random.default_rng(config["seed"])
    matrix = context.positive.astype(np.float64)
    n, m = matrix.shape
    if name == "item_knn_block":
        block = min(settings["ease_solve_block_size"], m)
        def run():
            cooccurrence = (matrix.T @ matrix[:, :block]).toarray()
            norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
            denominator = norms[:, None] * norms[None, :block]
            similarity = np.divide(cooccurrence, denominator, out=np.zeros_like(cooccurrence), where=denominator > 0)
            similarity *= cooccurrence / (cooccurrence + 10.)
            similarity[np.arange(block), np.arange(block)] = 0
            selected = np.argpartition(similarity, m - min(50, m), axis=0)[-min(50, m):]
            return {"finite": bool(np.isfinite(similarity).all()), "block_items": block,
                    "selected_neighbors": int(selected.size), "iteration_kind": "one_item_block"}
        return run, 0
    if name in {"ials_sweep", "custom_ials_sweep"}:
        preferences = matrix.copy()
        if name == "custom_ials_sweep":
            positive = history.loc[history["rating"].ge(config["evaluation"]["relevance_rating_threshold"])]
            rows = pd.Index(context.user_ids).get_indexer(positive["user_id"])
            columns = pd.Index(context.item_ids).get_indexer(positive["item_id"])
            age = (context.cutoff - positive["timestamp"]).dt.total_seconds().to_numpy() / 86400
            target = ((positive["rating"].to_numpy() - 3) / 2) * np.exp2(-age / 730.)
            preferences = sparse.coo_matrix((target, (rows, columns)), shape=matrix.shape).tocsr()
        transpose, pref_transpose = matrix.T.tocsr(), preferences.T.tocsr()
        user_factors, item_factors = rng.normal(0, .01, (n, rank)), rng.normal(0, .01, (m, rank))
        def update(observed, target, fixed, output):
            base = fixed.T @ fixed + regularization * np.eye(rank)
            for row in range(observed.shape[0]):
                start, end = observed.indptr[row:row + 2]
                indices = observed.indices[start:end]
                factors = fixed[indices]
                a = base + alpha * (factors.T @ factors)
                b = factors.T @ ((1 + alpha) * target.data[target.indptr[row]:target.indptr[row + 1]])
                output[row] = np.linalg.solve(a, b)
        def run():
            update(matrix, preferences, item_factors, user_factors)
            update(transpose, pref_transpose, user_factors, item_factors)
            return {"finite": bool(np.isfinite(user_factors).all() and np.isfinite(item_factors).all()),
                    "rank": rank, "iteration_kind": "one_alternating_sweep"}
        return run, 0
    if name == "ease_fragment":
        block = min(settings["ease_solve_block_size"], m)
        def run():
            gram = (matrix.T @ matrix).toarray(order="F")
            gram[np.diag_indices(m)] += 100.
            start = time.perf_counter()
            factor, lower = linalg.cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
            factor_seconds = time.perf_counter() - start
            shared = np.shares_memory(factor, gram)
            del gram
            weights = np.empty((m, m), dtype=np.float64, order="F")
            start = time.perf_counter()
            for left in range(0, m, block):
                columns = np.arange(left, min(left + block, m))
                rhs = np.zeros((m, len(columns)), dtype=np.float64, order="F")
                rhs[columns, np.arange(len(columns))] = 1
                inverse_columns = linalg.cho_solve((factor, lower), rhs, overwrite_b=True, check_finite=False)
                diagonal = inverse_columns[columns, np.arange(len(columns))].copy()
                inverse_columns /= -diagonal
                weights[:, columns] = inverse_columns
                weights[columns, columns] = 0
            solve_seconds = time.perf_counter() - start
            return {"finite": bool(np.isfinite(weights).all()), "diagonal_zero": bool((np.diag(weights) == 0).all()),
                    "factor_reused_gram_buffer": bool(shared), "cholesky_seconds": factor_seconds,
                    "blocked_solve_seconds": solve_seconds, "iteration_kind": "one_full_fit_on_fragment"}
        return run, 0
    if name == "fm_minibatch":
        import torch
        from utils.eda import build_eda_profile
        profile = build_eda_profile(history, config)
        numeric = np.column_stack([profile["event_metadata"][entity][config["features"][entity]].to_numpy()
                                   for entity in ("user", "item")]).astype(np.float32)
        count = min(settings["fm_batch_size"], len(history))
        users = pd.Index(context.user_ids).get_indexer(history["user_id"])
        items = pd.Index(context.item_ids).get_indexer(history["item_id"])
        ids = np.column_stack([users[:count], n + items[:count],
                               np.broadcast_to(n + m + np.arange(numeric.shape[1]), (count, numeric.shape[1]))])
        values = np.column_stack([np.ones((count, 2), dtype=np.float32), numeric[:count]])
        ids, values = torch.tensor(ids), torch.tensor(values)
        target = torch.tensor(history["rating"].iloc[:count].ge(config["evaluation"]["relevance_rating_threshold"]).to_numpy(dtype=np.float32))
        size = n + m + numeric.shape[1]
        factors = torch.nn.Parameter(torch.randn(size, rank) * .01)
        linear, bias = torch.nn.Parameter(torch.zeros(size)), torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([factors, linear, bias], lr=.001)
        def run():
            optimizer.zero_grad(set_to_none=True)
            interactions = factors[ids] * values[..., None]
            logits = bias + (linear[ids] * values).sum(1) + .5 * (
                interactions.sum(1).square() - interactions.square().sum(1)
            ).sum(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target) + 1e-4 * factors.square().mean()
            loss.backward()
            optimizer.step()
            return {"finite": bool(torch.isfinite(loss).item()), "batch_size": count, "rank": rank,
                    "negative_sampling_in_probe": False, "iteration_kind": "one_minibatch_after_warmup"}
        return run, 1
    raise ValueError(f"Unknown kernel: {name}")


def _worker(name: str, input_path: str, config: dict[str, Any]) -> dict[str, Any]:
    configure_reproducibility(config)
    start = time.perf_counter()
    frame = pd.read_csv(input_path, dtype={"user_id": "string", "item_id": "string"}, parse_dates=["timestamp"])
    context = build_history_context(frame, config["split"]["validation_start_utc"], config)
    run, warmups = _prepare_kernel(name, frame, context, config)
    for _ in range(warmups):
        run()
    setup = time.perf_counter() - start
    baseline = psutil.Process().memory_info().rss
    start = time.perf_counter()
    details = run()
    elapsed = time.perf_counter() - start
    # Linux getrusage can inherit a pre-exec parent high-water mark. VmHWM is
    # reset at exec and measures this worker's resident memory accurately.
    status = Path("/proc/self/status")
    if not status.exists():
        raise RuntimeError("Peak RSS probes currently require Linux /proc/self/status")
    peak = next(int(line.split()[1]) * 1024 for line in status.read_text().splitlines() if line.startswith("VmHWM:"))
    return {"kernel": name, "one_iteration_seconds": elapsed, "setup_and_warmup_seconds": setup,
            "baseline_rss_bytes": baseline, "peak_process_rss_bytes": peak, "warmup_iterations": warmups,
            "n_users": len(context.user_ids), "n_items": len(context.item_ids), "n_interactions": len(frame),
            "cpu_threads": config["runtime"]["cpu_threads"], **details}


def run_resource_checks(train: pd.DataFrame, refit_history: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run kernels sequentially in fresh processes; never access test observations."""
    directory = resolve_data_dir(config) / "resource_probe"
    directory.mkdir(parents=True, exist_ok=True)
    fragment = select_probe_history(train, config)
    payload = fragment.to_csv(index=False, lineterminator="\n").encode("utf-8")
    fragment_path = directory / "train_fragment.csv.gz"
    with fragment_path.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            zipped.write(payload)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    available_start = psutil.virtual_memory().available
    rows = []
    for kernel in ("item_knn_block", "ials_sweep", "ease_fragment", "fm_minibatch", "custom_ials_sweep"):
        process = subprocess.run(
            [sys.executable, "-m", "utils.resources", "--worker", kernel, "--input", str(fragment_path), "--config", str(config_path)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=config["runtime"]["resource_probe"]["worker_timeout_seconds"],
            env={**os.environ, "PYTHONHASHSEED": str(config["seed"])},
        )
        if process.returncode:
            raise RuntimeError(f"Resource probe {kernel} failed: {process.stderr[-2000:]}")
        row = json.loads(process.stdout)
        if not row["finite"]:
            raise ValueError(f"Nonfinite resource probe output: {kernel}")
        rows.append(row)
    available = min(available_start, psutil.virtual_memory().available)
    train_context = build_history_context(train, config["split"]["validation_start_utc"], config)
    refit_context = build_history_context(refit_history, config["split"]["test_start_utc"], config)
    estimates = {
        "train": estimate_ease_memory(train_context, config, available),
        "train_validation": estimate_ease_memory(refit_context, config, available),
    }
    ease_probe = next(row for row in rows if row["kernel"] == "ease_fragment")
    if not ease_probe["factor_reused_gram_buffer"]:
        raise ValueError("Observed Cholesky copied the Gram matrix; revise the full-size memory estimate")
    for estimate in estimates.values():
        ratio = estimate["n_history_items"] / ease_probe["n_items"]
        estimate["rough_dense_time_seconds_cubic_extrapolation"] = (
            ease_probe["cholesky_seconds"] + ease_probe["blocked_solve_seconds"]
        ) * ratio**3
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    report = {
        "scope": "resource_kernels_only_not_full_model_training_or_quality",
        "probe_source": "train_only", "probe_selection": "top_positive_items_then_seeded_positive_users",
        "probe_content_sha256": hashlib.sha256(payload).hexdigest(), "seed": config["seed"],
        "settings": config["runtime"]["resource_probe"],
        "environment": {"python": platform.python_version(), "cpu": cpu, "cpu_threads": config["runtime"]["cpu_threads"],
                        "available_bytes_at_start": available_start,
                        "packages": {name: metadata.version(name) for name in ("numpy", "scipy", "torch")}},
        "kernel_probes": rows, "ease_memory_estimates": estimates,
        "full_catalog_feasible": all(estimate["fits_optimized_strategy"] for estimate in estimates.values()),
        "memory_measurement": "Linux VmHWM after exec: peak RSS of each isolated worker, including imports, input, setup, and warmup",
        "time_extrapolation_is_measured_full_training": False,
        "execution_plan": "sequential_models_inplace_cholesky_block_solves_recheck_memory_before_each_fit",
        "test_labels_used": False,
    }
    results = PROJECT_ROOT / config["storage"]["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    (results / "resource_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(results / "resource_probe.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--worker", default=None)
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.worker:
        print(json.dumps(_worker(args.worker, args.input, config)))
        return
    configure_reproducibility(config)
    raw, _ = obtain_reviews(config)
    cleaned, _ = clean_reviews(raw, config)
    periods, _ = temporal_split(cleaned, config)
    report = run_resource_checks(periods["train"], pd.concat([periods["train"], periods["validation"]]), config)
    print(pd.DataFrame(report["kernel_probes"])[["kernel", "one_iteration_seconds", "peak_process_rss_bytes"]].to_string(index=False))
    print(json.dumps(report["ease_memory_estimates"], indent=2))
    if not report["full_catalog_feasible"]:
        raise RuntimeError("Full EASE allocation exceeds the current budget; revise execution before model search")


if __name__ == "__main__":
    main()
