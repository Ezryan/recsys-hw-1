"""Global temporal splits with fixed UTC cutoffs and no label-based selection.

Candidate sets and cold-start cohorts are implemented in utils.evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.cleaning import CORE_COLUMNS, clean_reviews
from utils.data import obtain_reviews
from utils.reproducibility import PROJECT_ROOT, load_config, resolve_data_dir


BOUNDARY_POLICY = "nearest_cumulative_count_train_then_train_validation"


def _validate_timestamps(timestamps: pd.Series) -> None:
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError("Timestamps must be timezone-aware UTC datetimes")
    if str(timestamps.dt.tz) != "UTC" or timestamps.isna().any():
        raise ValueError("Timestamps must be non-null and in UTC")


def choose_temporal_boundaries(
    timestamps: pd.Series, quantiles: list[float] | tuple[float, float]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Choose observed timestamps using event counts, without accessing labels.

    First choose the group boundary closest to q1*N, reserving at least one
    subsequent boundary. Then choose the closest boundary to q2*N strictly
    after the first. Ties select the earlier timestamp. At least three timestamp
    groups are required, so every period is nonempty even for large tied groups.
    """
    _validate_timestamps(timestamps)
    if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("Expected two increasing quantiles strictly between 0 and 1")
    counts = timestamps.value_counts().sort_index()
    if len(counts) < 3:
        raise ValueError("At least three distinct timestamps are required for three periods")
    counts_before = counts.cumsum().to_numpy()[:-1]
    first = int(np.abs(counts_before[:-1] - quantiles[0] * len(timestamps)).argmin())
    second = first + 1 + int(
        np.abs(counts_before[first + 1:] - quantiles[1] * len(timestamps)).argmin()
    )
    return counts.index[first + 1], counts.index[second + 1]


def split_content_sha256(frame: pd.DataFrame) -> str:
    """Identify the cleaned input; the hash is never used to choose cutoffs."""
    ordered = frame.sort_values(["timestamp", "source_line"], kind="stable")
    payload = ordered[CORE_COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def temporal_split(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Apply frozen cutoffs and report counts/time ranges, not held-out labels."""
    policy = config["split"]
    if policy["method"] != "global_temporal" or policy["timezone"] != "UTC":
        raise ValueError("Only the documented global temporal UTC split is supported")
    if not policy["equal_timestamps_stay_together"]:
        raise ValueError("Equal timestamps must stay in one period")
    missing = set(CORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Required cleaned columns are missing: {sorted(missing)}")
    _validate_timestamps(frame["timestamp"])
    if not frame["source_line"].is_unique:
        raise ValueError("Source lines must be unique")
    starts = [policy["validation_start_utc"], policy["test_start_utc"]]
    if any(value is None for value in starts):
        raise ValueError("Temporal cutoffs are not fixed; run with --freeze-config first")
    t1, t2 = (pd.Timestamp(value) for value in starts)
    if t1.tzinfo is None or t2.tzinfo is None:
        raise ValueError("Frozen cutoffs must have explicit UTC offsets")
    t1, t2 = t1.tz_convert("UTC"), t2.tz_convert("UTC")
    if not t1 < t2:
        raise ValueError("Validation cutoff must precede test cutoff")
    frame = frame.sort_values(["timestamp", "source_line"], kind="stable").reset_index(drop=True)
    content_hash = split_content_sha256(frame)
    expected_hash = policy.get("boundary_source_content_sha256")
    if expected_hash is not None and expected_hash != content_hash:
        raise ValueError("Cleaned input differs from the dataset used to fix cutoffs")
    times = frame["timestamp"]
    splits = {
        "train": frame.loc[times < t1].copy(),
        "validation": frame.loc[(times >= t1) & (times < t2)].copy(),
        "test": frame.loc[times >= t2].copy(),
    }
    if any(part.empty for part in splits.values()):
        raise ValueError("Every temporal period must be nonempty")
    membership = _membership(splits)
    checks = {
        "periods_nonempty": all(not part.empty for part in splits.values()),
        "all_rows_assigned_once": len(membership) == len(frame) and membership["source_line"].is_unique,
        "strict_temporal_order": bool(
            splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
            and splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()
        ),
        "equal_timestamps_stay_together": bool(pd.concat([
            part[["timestamp"]].assign(period=name) for name, part in splits.items()
        ]).groupby("timestamp")["period"].nunique().eq(1).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Temporal split invariants failed: {checks}")
    membership_bytes = membership.to_csv(index=False, lineterminator="\n").encode("utf-8")
    report = {
        "method": policy["method"],
        "timezone": "UTC",
        "boundary_selection": policy.get("boundary_selection", BOUNDARY_POLICY),
        "target_cumulative_fractions": policy["boundary_quantiles"],
        "validation_start_utc": t1.isoformat(),
        "test_start_utc": t2.isoformat(),
        "cleaned_content_sha256": content_hash,
        "membership_sha256": hashlib.sha256(membership_bytes).hexdigest(),
        "n_interactions": len(frame),
        "periods": {
            name: {
                "n_interactions": len(part),
                "fraction": len(part) / len(frame),
                "timestamp_min_utc": part["timestamp"].min().isoformat(),
                "timestamp_max_utc": part["timestamp"].max().isoformat(),
            } for name, part in splits.items()
        },
        "checks": checks,
        "evaluation_protocol": dict(config["evaluation"]),
        "held_out_rating_statistics_included": False,
    }
    return splits, report


def _membership(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([
        part[["source_line"]].assign(period=name) for name, part in splits.items()
    ], ignore_index=True).sort_values("source_line", kind="stable").reset_index(drop=True)


def save_split_report(
    splits: dict[str, pd.DataFrame], report: dict[str, Any], config: dict[str, Any]
) -> Path:
    """Keep row-level membership local; publish only aggregates and hashes."""
    membership = _membership(splits).to_csv(index=False, lineterminator="\n").encode("utf-8")
    if hashlib.sha256(membership).hexdigest() != report["membership_sha256"]:
        raise ValueError("Split membership differs from its report")
    data_dir = resolve_data_dir(config)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "temporal_membership.csv.gz"
    with path.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(membership)
    results_dir = PROJECT_ROOT / config["storage"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "split_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--freeze-config", action="store_true", help="Fix missing cutoffs from timestamp counts")
    args = parser.parse_args()
    config = load_config(args.config)
    raw, _ = obtain_reviews(config)
    frame, _ = clean_reviews(raw, config)
    policy = config["split"]
    starts = [policy["validation_start_utc"], policy["test_start_utc"]]
    if args.freeze_config and all(value is None for value in starts):
        t1, t2 = choose_temporal_boundaries(frame["timestamp"], policy["boundary_quantiles"])
        policy.update({
            "validation_start_utc": t1.isoformat(),
            "test_start_utc": t2.isoformat(),
            "boundary_selection": BOUNDARY_POLICY,
            "boundary_source_content_sha256": split_content_sha256(frame),
        })
        config["stage"] = "temporal_protocol_fixed"
    splits, report = temporal_split(frame, config)
    path = save_split_report(splits, report, config)
    if args.freeze_config:
        config_path = PROJECT_ROOT / args.config
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame.from_dict(report["periods"], orient="index").to_string())
    print(f"All temporal checks passed; local membership: {path}")


if __name__ == "__main__":
    main()
