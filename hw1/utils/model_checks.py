"""Fixed-parameter implementation checks on train/validation, without search."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import pandas as pd
import numpy as np
from scipy import sparse

from models import BaseRecommender, ItemKNN, IALS, EASE, FM, CustomIALS
from models.custom_ials import custom_preferences, summarize_targets, validate_target_params
from utils.cleaning import clean_reviews
from utils.data import obtain_reviews
from utils.evaluation import EvaluationStage, build_evaluation_stage
from utils.reproducibility import PROJECT_ROOT, configure_reproducibility, load_config, resolve_data_dir
from utils.splitting import temporal_split


def run_implementation_check(
    model: BaseRecommender, history: pd.DataFrame, stage: EvaluationStage,
    config: dict[str, Any], name: str,
) -> dict[str, Any]:
    """Fit once, evaluate one common validation cohort, save traceable artifacts."""
    if stage.period != "validation":
        raise ValueError("Implementation checks use validation; final test remains held out")
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
        raise ValueError("Model name must be a simple lowercase artifact identifier")
    start = time.perf_counter()
    model.fit(history, config, context=stage.context)
    fit_total = time.perf_counter() - start
    start = time.perf_counter()
    result = model.evaluate(stage)
    evaluation_seconds = time.perf_counter() - start
    report = {
        "model": name, "purpose": "implementation_check_fixed_parameters_no_search",
        "hyperparameters_tuned": False, "params": model.params,
        "seed": config["seed"], "cpu_threads": config["runtime"]["cpu_threads"],
        "fit_total_seconds": fit_total, "fit_body_seconds": model.fit_seconds_,
        "fit_reused_cache": bool(getattr(model, "cache_hit_", False)),
        "fit_compute_seconds": getattr(model, "training_compute_seconds_", model.fit_seconds_),
        "evaluation_seconds": evaluation_seconds,
        "validation": result.summary, "training_details": getattr(model, "training_summary_", {}),
        "test_used": False,
    }
    local_dir = resolve_data_dir(config) / "model_checks" / name
    local_dir.mkdir(parents=True, exist_ok=True)
    result.per_user.to_csv(local_dir / "validation_per_user.csv.gz", index=False,
                           compression={"method": "gzip", "mtime": 0})
    if sparse.issparse(getattr(model, "similarity_", None)):
        sparse.save_npz(local_dir / "similarity.npz", model.similarity_)
    if hasattr(model, "user_factors_") and hasattr(model, "item_factors_"):
        np.save(local_dir / "user_factors.npy", model.user_factors_)
        np.save(local_dir / "item_factors.npy", model.item_factors_)
    if hasattr(model, "save_artifacts"):
        model.save_artifacts(local_dir)
    results = PROJECT_ROOT / config["storage"]["results_dir"]
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{name}_implementation_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame([{
        "model": name, "purpose": report["purpose"], "params": json.dumps(model.params, sort_keys=True),
        "validation_ndcg_at_10": result.summary["ndcg@10"], "validation_recall_at_10": result.summary["recall@10"],
        "fit_total_seconds": fit_total, "fit_body_seconds": model.fit_seconds_,
        "fit_reused_cache": report["fit_reused_cache"], "fit_compute_seconds": report["fit_compute_seconds"],
        "evaluation_seconds": evaluation_seconds, "n_users": result.summary["n_users"],
        "stage_sha256": result.summary["stage_sha256"],
    }]).to_csv(results / f"{name}_implementation_check.csv", index=False)
    if hasattr(model, "loss_history_"):
        pd.DataFrame(model.loss_history_).to_csv(results / f"{name}_training_objective.csv", index=False)
    return report


def run_fm_control_checks(history, stage, config):
    """One fixed setting, same samples/epochs/initial ID weights for both FMs."""
    params = config["implementation_checks"]["fm"]
    primary, control = FM(**params), FM(**{**params, "use_metadata": False})
    main_report = run_implementation_check(primary, history, stage, config, "fm")
    control_report = run_implementation_check(control, history, stage, config, "fm_ids_only")
    assert main_report["training_details"]["examples_sha256"] == control_report["training_details"]["examples_sha256"]
    rows = []
    for report in (main_report, control_report):
        rows.append({"model": report["model"], "use_metadata": report["params"]["use_metadata"],
                     "ndcg@10": report["validation"]["ndcg@10"], "recall@10": report["validation"]["recall@10"],
                     "epochs": report["params"]["epochs"], "n_examples": report["training_details"]["n_examples"],
                     "examples_sha256": report["training_details"]["examples_sha256"],
                     "stage_sha256": report["validation"]["stage_sha256"]})
    table = pd.DataFrame(rows)
    results = PROJECT_ROOT / config["storage"]["results_dir"]
    table.to_csv(results / "fm_metadata_control.csv", index=False)
    return primary, control, main_report, control_report


def run_custom_ials_control_checks(history, stage, config):
    """Fixed baseline, rating-only, recency-only, and combined target; no tuning."""
    structural = config["implementation_checks"]["ials"]
    target = validate_target_params(**config["implementation_checks"]["custom_ials"])
    if target["gamma"] <= 0 or target["half_life_days"] == "infinity":
        raise ValueError("The primary custom check must use both rating and finite recency")
    models = {
        "binary": IALS(**structural),
        "rating_only": CustomIALS(**structural, gamma=target["gamma"], half_life_days="infinity"),
        "recency_only": CustomIALS(**structural, gamma=0, half_life_days=target["half_life_days"]),
        "rating_and_recency": CustomIALS(**structural, **target),
    }
    names = {"binary": "ials", "rating_only": "custom_ials_rating_only",
             "recency_only": "custom_ials_recency_only", "rating_and_recency": "custom_ials"}
    reports, rows, distributions = {}, [], []
    for variant, model in models.items():
        logging.getLogger(__name__).info("Custom-target check: %s", variant)
        report = run_implementation_check(model, history, stage, config, names[variant])
        reports[variant] = report
        baseline = models["binary"]
        if (not np.array_equal(baseline.observed_.indptr, model.observed_.indptr)
                or not np.array_equal(baseline.observed_.indices, model.observed_.indices)
                or not np.array_equal(baseline.observed_.data, model.observed_.data)):
            raise ValueError("Custom-target controls changed the observed confidence pattern")
        rows.append({
            "variant": variant, "model": names[variant],
            **structural, "gamma": model.params.get("gamma", 0.), "half_life_days": model.params.get("half_life_days", "infinity"),
            "ndcg@10": report["validation"]["ndcg@10"], "recall@10": report["validation"]["recall@10"],
            "ndcg_delta_vs_binary": report["validation"]["ndcg@10"] - reports["binary"]["validation"]["ndcg@10"],
            "recall_delta_vs_binary": report["validation"]["recall@10"] - reports["binary"]["validation"]["recall@10"],
            "fit_seconds": report["fit_total_seconds"], "evaluation_seconds": report["evaluation_seconds"],
            "stage_sha256": report["validation"]["stage_sha256"],
            "observed_pattern_sha256": report["training_details"]["observed_pattern_sha256"],
            "target_sha256": report["training_details"]["target_sha256"],
        })
        distributions.append({"variant": variant, **summarize_targets(model.targets_.data)})
    results = PROJECT_ROOT / config["storage"]["results_dir"]
    pd.DataFrame(rows).to_csv(results / "custom_ials_target_control.csv", index=False)
    pd.DataFrame(distributions).to_csv(results / "custom_ials_target_distribution.csv", index=False)
    # Grouped examples from train; publish no entity IDs or individual events.
    positive = history.loc[history["rating"].ge(4)].copy()
    positive["age_days"] = (stage.context.cutoff - positive["timestamp"]).dt.total_seconds() / 86400
    positive["age_band_days"] = pd.cut(positive["age_days"], [0, 180, 730, np.inf], labels=["(0,180]", "(180,730]", "(730,infinity)"])
    for variant in ("rating_only", "recency_only", "rating_and_recency"):
        params = models[variant].params
        positive[variant] = custom_preferences(positive["rating"], positive["timestamp"], stage.context.cutoff,
                                              gamma=params["gamma"], half_life_days=params["half_life_days"])
    grouped = positive.groupby(["rating", "age_band_days"], observed=True).agg(
        n_positive_reviews=("rating", "size"), median_age_days=("age_days", "median"),
        rating_only_mean=("rating_only", "mean"), recency_only_mean=("recency_only", "mean"),
        custom_target_mean=("rating_and_recency", "mean"), custom_target_min=("rating_and_recency", "min"),
        custom_target_max=("rating_and_recency", "max"),
    ).reset_index()
    grouped.to_csv(results / "custom_ials_target_by_rating_age.csv", index=False)
    return models, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--model", choices=["item_knn", "ials", "ease", "fm", "custom_ials"], default="item_knn")
    parser.add_argument("--no-cache", action="store_true", help="Recompute weights and refresh cache")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.no_cache:
        config["runtime"]["reuse_model_cache"] = False
    if args.model in {"ease", "fm", "custom_ials"}:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    configure_reproducibility(config)
    raw, _ = obtain_reviews(config)
    cleaned, _ = clean_reviews(raw, config)
    periods, _ = temporal_split(cleaned, config)
    stage = build_evaluation_stage(periods, config, "validation")
    if args.model == "fm":
        _, _, primary, control = run_fm_control_checks(periods["train"], stage, config)
        print(json.dumps({"fm": primary, "fm_ids_only": control}, ensure_ascii=False, indent=2))
        return
    if args.model == "custom_ials":
        _, reports = run_custom_ials_control_checks(periods["train"], stage, config)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return
    model_class = {"item_knn": ItemKNN, "ials": IALS, "ease": EASE}[args.model]
    model = model_class(**config["implementation_checks"][args.model])
    report = run_implementation_check(model, periods["train"], stage, config, args.model)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
