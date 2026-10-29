"""Deterministic review cleaning with an audit and assignment size checks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from utils.data import obtain_reviews, parse_review_timestamps
from utils.reproducibility import PROJECT_ROOT, load_config, resolve_data_dir


CORE_COLUMNS = ["user_id", "item_id", "rating", "timestamp", "source_line"]


def _freeze_metadata(value: Any) -> Any:
    """Make nested JSON values hashable; dictionary key order is immaterial."""
    if isinstance(value, dict):
        return ("dict", tuple(sorted((key, _freeze_metadata(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_freeze_metadata(item) for item in value))
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def validate_cleaned_reviews(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Fail explicitly on invalid output or a sample outside assignment limits."""
    if set(CORE_COLUMNS) - set(frame.columns):
        raise ValueError("Cleaned data is missing required columns")
    for column in ("user_id", "item_id"):
        values = frame[column]
        if values.isna().any() or not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain non-null original string IDs")
        if values.str.strip().eq("").any():
            raise ValueError(f"{column} contains an empty ID")
    if not pd.api.types.is_numeric_dtype(frame["rating"]):
        raise ValueError("Ratings must be numeric")
    ratings = frame["rating"]
    if not (np.isfinite(ratings) & ratings.between(
        config["cleaning"]["rating_min"], config["cleaning"]["rating_max"]
    )).all():
        raise ValueError("Ratings outside the valid range")
    if not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError("Timestamps must be timezone-aware UTC datetimes")
    if str(frame["timestamp"].dt.tz) != "UTC" or frame["timestamp"].isna().any():
        raise ValueError("Timestamps must be valid UTC datetimes")
    lines = frame["source_line"]
    if not pd.api.types.is_integer_dtype(lines) or not lines.is_unique or not lines.gt(0).all():
        raise ValueError("Source lines must be unique positive integers")
    if frame.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Repeated user-item pairs remain after cleaning")
    ordered = frame.sort_values(["timestamp", "source_line"], kind="stable")
    if not frame[CORE_COLUMNS].equals(ordered[CORE_COLUMNS]):
        raise ValueError("Cleaned data must be ordered by timestamp and source line")

    limits = config["dataset"]["constraints"]
    actual = {
        "interactions": len(frame),
        "users": int(frame["user_id"].nunique()),
        "items": int(frame["item_id"].nunique()),
    }
    checks = {
        "interactions": {"actual": actual["interactions"], "min": limits["min_interactions"], "max": limits["max_interactions"]},
        "users": {"actual": actual["users"], "min": limits["min_users"]},
        "items": {"actual": actual["items"], "min": limits["min_items"]},
    }
    for name, check in checks.items():
        check["passed"] = check["actual"] >= check["min"] and (
            "max" not in check or check["actual"] <= check["max"]
        )
        if not check["passed"]:
            raise ValueError(f"Cleaned sample violates assignment constraints for {name}: {check}")
    return checks


def clean_reviews(raw: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep the first valid interaction per pair and its original rating.

    The caller's raw frame is untouched. Optional metadata is retained and never
    used to filter rows. No k-core, positivity filter, or subsampling is applied.
    """
    policy = config["cleaning"]
    supported = {
        "drop_missing_ids": True,
        "drop_invalid_timestamps": True,
        "drop_exact_duplicates": True,
        "pair_duplicates": "earliest_timestamp_then_source_line",
        "keep_low_ratings_in_history": True,
        "additional_k_core": None,
        "subsample": None,
    }
    if any(policy.get(key) != value for key, value in supported.items()):
        raise ValueError("Unsupported cleaning policy; implement and document it before use")
    mapping = {source: canonical for canonical, source in config["dataset"]["fields"].items()}
    missing = (set(mapping) | {"source_line"}) - set(raw.columns)
    if missing:
        raise ValueError(f"Required source columns are missing: {sorted(missing)}")
    if any(target in raw.columns and target != source for source, target in mapping.items()):
        raise ValueError("A canonical column name collides with an original field")
    lines = raw["source_line"]
    if not pd.api.types.is_integer_dtype(lines) or not lines.is_unique or not lines.gt(0).all():
        raise ValueError("Source lines must be unique positive integers")

    source = raw.sort_values("source_line", kind="stable").reset_index(drop=True)
    frame = source.rename(columns=mapping).copy()
    for name in ("user_id", "item_id"):
        frame[name] = frame[name].astype("string")
    audit = []

    def record(stage: str, before: int) -> None:
        audit.append({
            "stage": stage, "input_rows": before, "output_rows": len(frame),
            "removed_rows": before - len(frame),
            "n_users": int(frame["user_id"].nunique()),
            "n_items": int(frame["item_id"].nunique()),
        })

    record("input", len(source))
    before = len(frame)
    valid_ids = frame["user_id"].notna() & frame["item_id"].notna()
    valid_ids &= frame["user_id"].str.strip().ne("") & frame["item_id"].str.strip().ne("")
    frame = frame.loc[valid_ids].copy()
    record("drop_missing_ids", before)

    before = len(frame)
    dates = parse_review_timestamps(frame["timestamp"])
    invalid_boolean = frame["timestamp"].map(lambda value: isinstance(value, (bool, np.bool_)))
    frame["timestamp"] = dates
    frame = frame.loc[dates.notna() & ~invalid_boolean].copy()
    record("drop_invalid_timestamps", before)

    before = len(frame)
    invalid_boolean = frame["rating"].map(lambda value: isinstance(value, (bool, np.bool_)))
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce").astype("float64")
    valid_ratings = np.isfinite(frame["rating"]) & frame["rating"].between(policy["rating_min"], policy["rating_max"])
    frame = frame.loc[valid_ratings & ~invalid_boolean].copy()
    record("drop_invalid_ratings", before)

    # Compare original fields, excluding only the added source_line. Metadata
    # may contain dicts/lists, which pandas cannot hash without normalization.
    before = len(frame)
    comparison = source.loc[frame.index].drop(columns="source_line").copy()
    for column in comparison.columns:
        if comparison[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            comparison[column] = comparison[column].map(_freeze_metadata)
    frame = frame.loc[~comparison.duplicated(keep="first")].copy()
    record("drop_exact_duplicates", before)

    before = len(frame)
    frame = frame.sort_values(["timestamp", "source_line"], kind="stable")
    frame = frame.drop_duplicates(["user_id", "item_id"], keep="first").reset_index(drop=True)
    record("keep_earliest_pair", before)
    checks = validate_cleaned_reviews(frame, config)
    threshold = config["evaluation"]["relevance_rating_threshold"]
    core_bytes = frame[CORE_COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    summary = {
        "stage": "cleaned_before_split",
        "source_archive_sha256": config["dataset"]["sha256"],
        "content_sha256": hashlib.sha256(core_bytes).hexdigest(),
        "n_interactions": len(frame),
        "n_users": int(frame["user_id"].nunique()),
        "n_items": int(frame["item_id"].nunique()),
        "n_positive_interactions": int(frame["rating"].ge(threshold).sum()),
        "n_low_rating_interactions": int(frame["rating"].lt(threshold).sum()),
        "timestamp_min_utc": frame["timestamp"].min().isoformat(),
        "timestamp_max_utc": frame["timestamp"].max().isoformat(),
        "dtypes": {name: str(frame[name].dtype) for name in CORE_COLUMNS},
        "optional_field_missing_rows": {
            name: int(frame[name].isna().sum()) for name in sorted(set(frame.columns) - set(CORE_COLUMNS))
        },
        "cleaning_rules": dict(policy),
        "audit": audit,
        "constraints": checks,
        "cleaned_constraints_verified": True,
    }
    return frame, summary


def save_cleaned_reviews(frame: pd.DataFrame, summary: dict[str, Any], config: dict[str, Any]) -> Path:
    """Store core interactions locally; publish only aggregate audit/statistics."""
    validate_cleaned_reviews(frame, config)
    payload = frame[CORE_COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != summary["content_sha256"]:
        raise ValueError("Cleaned frame differs from the summary content hash")
    data_dir = resolve_data_dir(config)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "cleaned_interactions.csv.gz"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as stream:
            temporary_path = Path(stream.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
                compressed.write(payload)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    results_dir = PROJECT_ROOT / config["storage"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary["audit"]).to_csv(results_dir / "cleaning_audit.csv", index=False)
    (results_dir / "clean_data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    args = parser.parse_args()
    config = load_config(args.config)
    raw, _ = obtain_reviews(config)
    cleaned, summary = clean_reviews(raw, config)
    output_path = save_cleaned_reviews(cleaned, summary, config)
    print(pd.DataFrame(summary["audit"]).to_string(index=False))
    print(json.dumps({
        "n_interactions": summary["n_interactions"],
        "n_users": summary["n_users"],
        "n_items": summary["n_items"],
        "n_low_rating_interactions": summary["n_low_rating_interactions"],
        "cleaned_constraints_verified": summary["cleaned_constraints_verified"],
        "local_dataset": str(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
