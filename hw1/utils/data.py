"""Download and inspect the original reviews without cleaning or splitting them.

The compressed archive and its acquisition manifest stay in RECSYS_DATA_DIR.
Only aggregate statistics and provenance may be published in results/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from utils.reproducibility import PROJECT_ROOT, load_config, resolve_data_dir


def sha256_file(path: str | Path) -> str:
    """Hash compressed bytes, independently of gzip header or JSON formatting."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Read every JSON object, retaining all fields and 1-based source lines.

    Malformed JSON/gzip is an error, never a silently dropped interaction.
    Missing field values remain intact for the separate cleaning step.
    """
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(2) != b"\x1f\x8b":
            raise ValueError(f"Expected a gzip archive, not HTML/plain text: {path.name}")
    records = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for source_line, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at source line {source_line}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object at source line {source_line}")
                if "source_line" in record:
                    raise ValueError("Original data unexpectedly contains reserved source_line")
                record["source_line"] = source_line
                records.append(record)
    except (OSError, EOFError, UnicodeError) as error:
        raise ValueError(f"Corrupt gzip or invalid UTF-8: {path.name}") from error
    if not records:
        raise ValueError("The reviews archive contains no records")
    return pd.DataFrame.from_records(records)


def parse_review_timestamps(values: pd.Series) -> pd.Series:
    """Coerce invalid/out-of-range Unix seconds without overflowing conversion."""
    seconds = pd.to_numeric(values, errors="coerce")
    seconds = seconds.where(seconds.between(
        pd.Timestamp.min.value / 1_000_000_000,
        pd.Timestamp.max.value / 1_000_000_000,
    ))
    return pd.to_datetime(seconds, unit="s", utc=True, errors="coerce")


def summarize_reviews(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the original schema and values without removing observations."""
    fields = config["dataset"]["fields"]
    missing_columns = sorted(set(fields.values()) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Required source columns are missing: {missing_columns}")
    users = frame[fields["user_id"]]
    items = frame[fields["item_id"]]
    ratings = pd.to_numeric(frame[fields["rating"]], errors="coerce")
    dates = parse_review_timestamps(frame[fields["timestamp"]])
    valid_ratings = ratings.between(config["cleaning"]["rating_min"], config["cleaning"]["rating_max"])
    if dates.notna().sum() == 0:
        raise ValueError("No valid timestamps: temporal evaluation is impossible")
    if valid_ratings.sum() == 0:
        raise ValueError("No ratings in the configured rating range")

    checks = {}
    for canonical, raw in fields.items():
        checks[canonical] = {
            "source_field": raw,
            "non_null_rows": int(frame[raw].notna().sum()),
            "missing_rows": int(frame[raw].isna().sum()),
            "dtype": str(frame[raw].dtype),
        }
    return {
        "stage": "raw_before_cleaning",
        "n_interactions": len(frame),
        "n_users": int(users.nunique()),
        "n_items": int(items.nunique()),
        "timestamp_min_utc": dates.min().isoformat(),
        "timestamp_max_utc": dates.max().isoformat(),
        "rating_distribution": {
            format(float(value), "g"): int(count)
            for value, count in ratings.value_counts().sort_index().items()
        },
        "invalid_rating_rows": int((~valid_ratings).sum()),
        "invalid_timestamp_rows": int(dates.isna().sum()),
        "required_fields": checks,
        "optional_field_non_null_rows": {
            name: int(frame[name].notna().sum())
            for name in sorted(set(frame.columns) - set(fields.values()) - {"source_line"})
        },
        "source_line_is_unique": bool(frame["source_line"].is_unique),
        "cleaned_constraints_verified": False,
    }


def _verify_archive(path: Path, config: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    digest = sha256_file(path)
    expected = config["dataset"].get("sha256")
    if expected is not None and digest != expected:
        raise ValueError(f"Archive SHA-256 mismatch: expected {expected}, got {digest}")
    frame = load_reviews(path)
    summarize_reviews(frame, config)
    return digest, frame


def obtain_reviews(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse a verified archive, or download to a temporary file and publish it.

    Existing corrupt/mismatched files fail visibly and are not overwritten.
    Fresh files are renamed only after the full gzip/JSON and checksum checks.
    """
    dataset = config["dataset"]
    filename = dataset["filename"]
    if Path(filename).name != filename:
        raise ValueError("dataset.filename must be a filename, not a directory path")
    data_dir = resolve_data_dir(config)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / filename
    manifest_path = data_dir / f"{filename}.manifest.json"
    now = datetime.now(timezone.utc).isoformat()
    if path.exists():
        digest, frame = _verify_archive(path, config)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["sha256"] != digest:
                raise ValueError("Existing archive differs from its acquisition manifest")
        else:
            manifest = {
                "name": dataset["name"],
                "source_url": dataset["source_url"],
                "requested_url": dataset["download_url"],
                "resolved_url": None,
                "downloaded_at_utc": None,
                "acquisition": "existing_file_without_download_metadata",
                "filename": filename,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        manifest["last_verified_at_utc"] = now
        _write_json(manifest_path, manifest)
        return frame, manifest

    download = dataset.get("download", {})
    attempts = download.get("attempts", 3)
    timeout = download.get("timeout_seconds", 30)
    max_bytes = download.get("max_bytes", 104857600)
    if attempts < 1:
        raise ValueError("download.attempts must be at least 1")
    last_error: Exception | None = None
    for attempt in range(attempts):
        temporary_path = None
        try:
            request = Request(dataset["download_url"], headers={"User-Agent": "recsys-hw-1/0.1"})
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(f"Expected HTTP 200, received {response.status}")
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("The download URL returned HTML instead of reviews")
                expected_size = response.headers.get("Content-Length")
                with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".part", delete=False) as stream:
                    temporary_path = Path(stream.name)
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"Download exceeds configured limit: {max_bytes} bytes")
                        stream.write(chunk)
                if expected_size is not None and size != int(expected_size):
                    raise ValueError("Incomplete response: size differs from Content-Length")
                resolved_url = response.url
                last_modified = response.headers.get("Last-Modified")
            digest, frame = _verify_archive(temporary_path, config)
            manifest = {
                "name": dataset["name"],
                "source_url": dataset["source_url"],
                "requested_url": dataset["download_url"],
                "resolved_url": resolved_url,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "acquisition": "http_download",
                "filename": filename,
                "sha256": digest,
                "size_bytes": size,
                "server_last_modified": last_modified,
                "last_verified_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            temporary_path.replace(path)
            _write_json(manifest_path, manifest)
            return frame, manifest
        except HTTPError as error:
            last_error = error
            if error.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if attempt + 1 < attempts:
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download reviews after HTTP/network failure: {last_error}") from last_error


def save_dataset_reports(
    manifest: dict[str, Any], summary: dict[str, Any], config: dict[str, Any]
) -> None:
    """Publish only provenance and aggregates, never individual review records."""
    results_dir = PROJECT_ROOT / config["storage"]["results_dir"]
    _write_json(results_dir / "dataset_manifest.json", manifest)
    _write_json(results_dir / "raw_data_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    args = parser.parse_args()
    config = load_config(args.config)
    frame, manifest = obtain_reviews(config)
    summary = summarize_reviews(frame, config)
    save_dataset_reports(manifest, summary, config)
    print(json.dumps({
        "sha256": manifest["sha256"],
        "size_bytes": manifest["size_bytes"],
        "n_interactions": summary["n_interactions"],
        "n_users": summary["n_users"],
        "n_items": summary["n_items"],
        "timestamp_min_utc": summary["timestamp_min_utc"],
        "timestamp_max_utc": summary["timestamp_max_utc"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
