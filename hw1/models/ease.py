"""EASE with in-place Cholesky, blocked inverse columns, and validated local cache."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy import linalg, sparse

from models.base import BaseRecommender
from utils.reproducibility import PROJECT_ROOT
from utils.resources import estimate_ease_memory


LOGGER = logging.getLogger(__name__)


def ease_objective(
    interactions: sparse.csr_matrix, weights: np.ndarray, *, regularization: float, block_size: int = 128,
) -> dict[str, float]:
    """Compute ||R-RB||² + lambda*||B||² using target-column blocks."""
    if not sparse.isspmatrix_csr(interactions):
        raise ValueError("EASE objective expects CSR interactions")
    weights = np.asarray(weights)
    m = interactions.shape[1]
    if weights.shape != (m, m):
        raise ValueError("Weights must be square and match history item columns")
    if not np.isfinite(regularization) or regularization <= 0 or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("Positive regularization and integer block size are required")
    reconstruction, squared_weights = 0., 0.
    for start in range(0, m, block_size):
        stop = min(start + block_size, m)
        block = weights[:, start:stop]
        if not np.isfinite(block).all():
            raise ValueError("EASE weights must be finite")
        residual = interactions[:, start:stop].toarray() - interactions @ block
        reconstruction += float(np.einsum("ij,ij->", residual, residual))
        squared_weights += float(np.einsum("ij,ij->", block, block))
    penalty = regularization * squared_weights
    return {"reconstruction_error": reconstruction, "l2_penalty": penalty, "objective": reconstruction + penalty}


def _weights_sha256(weights: np.ndarray, block_size: int = 128) -> str:
    """Hash canonical float64 columns without allocating a second dense matrix."""
    digest = hashlib.sha256()
    for start in range(0, weights.shape[1], block_size):
        block = np.asarray(weights[:, start:start + block_size], dtype="<f8", order="F")
        digest.update(memoryview(block.ravel(order="F")).cast("B"))
    return digest.hexdigest()


class EASE(BaseRecommender):
    """min ||R-RB||² + lambda*||B||², diag(B)=0; B[source,target]."""

    def __init__(self, regularization: float = 100., solve_block_size: int = 128) -> None:
        super().__init__(**self._validated_params(regularization, solve_block_size))

    @staticmethod
    def _validated_params(regularization, solve_block_size):
        if isinstance(regularization, (bool, np.bool_)) or not np.isscalar(regularization):
            raise ValueError("regularization must be a finite positive scalar")
        regularization = float(regularization)
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if isinstance(solve_block_size, (bool, np.bool_)) or not isinstance(solve_block_size, (int, np.integer)) or solve_block_size < 1:
            raise ValueError("solve_block_size must be a positive integer")
        return {"regularization": regularization, "solve_block_size": int(solve_block_size)}

    def _cache_location(self, config: dict[str, Any]) -> tuple[Path, str]:
        sources = ("models/ease.py", "models/base.py", "utils/evaluation.py", "utils/resources.py")
        code_digest = hashlib.sha256()
        for name in sources:
            code_digest.update(name.encode())
            code_digest.update((PROJECT_ROOT / name).read_bytes())
        identity = {
            "history_context": self.context_.fingerprint, "raw_archive_sha256": config["dataset"]["sha256"],
            "params": self.params, "seed": config["seed"], "split": config["split"], "features": config["features"],
            "code_sha256": code_digest.hexdigest(),
            "environment": {"numpy": np.__version__, "scipy": scipy.__version__,
                            "python": platform.python_version(), "machine": platform.machine()},
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(config["storage"]["cache_dir"]).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        return root / "ease" / key, key

    def _restore_cache(self, directory: Path, key: str) -> bool:
        metadata_path, weights_path = directory / "metadata.json", directory / "weights.npy"
        if not metadata_path.exists() or not weights_path.exists():
            return False
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved["cache_key"] != key:
            raise ValueError("EASE cache identity mismatch; force a recomputation")
        weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
        m = len(self.context_.item_ids)
        if weights.shape != (m, m) or weights.dtype != np.dtype("float64") or not weights.flags.f_contiguous:
            raise ValueError("EASE cache has an incompatible shape, dtype, or layout")
        if _weights_sha256(weights) != saved["training_summary"]["weights_sha256"]:
            raise ValueError("EASE cache checksum mismatch; run with --no-cache to recompute")
        self.weights_ = weights
        self.training_summary_ = saved["training_summary"]
        self.training_compute_seconds_ = saved["training_compute_seconds"]
        LOGGER.info("EASE: reused validated cache %s", key[:12])
        return True

    def _store_cache(self, directory: Path, key: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy.part", delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, self.weights_, allow_pickle=False)
            temporary.replace(directory / "weights.npy")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        document = {"cache_key": key, "training_summary": self.training_summary_,
                    "training_compute_seconds": self.training_compute_seconds_}
        temporary_metadata = directory / "metadata.json.part"
        temporary_metadata.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary_metadata.replace(directory / "metadata.json")

    def _fit(self, history: pd.DataFrame, config: dict[str, Any]) -> None:
        self.params = self._validated_params(**self.params)
        m = len(self.context_.item_ids)
        block_size = min(self.params["solve_block_size"], m)
        memory_config = deepcopy(config)
        memory_config["runtime"]["resource_probe"]["ease_solve_block_size"] = block_size
        memory_check = estimate_ease_memory(self.context_, memory_config, psutil.virtual_memory().available)
        if not memory_check["fits_optimized_strategy"]:
            raise MemoryError(f"EASE extra arrays need {memory_check['estimated_extra_peak_bytes']} bytes; budget is {memory_check['effective_extra_memory_budget_bytes']}")
        self.interactions_ = self.context_.positive.astype(np.float64)
        self.cache_path_, cache_key = self._cache_location(config)
        self.cache_hit_ = False
        if config["runtime"].get("reuse_model_cache", True):
            self.cache_hit_ = self._restore_cache(self.cache_path_, cache_key)
        if not self.cache_hit_:
            start = time.perf_counter()
            LOGGER.info("EASE: building Gram, %d items, lambda=%g", m, self.params["regularization"])
            sparse_gram = (self.interactions_.T @ self.interactions_).tocsc()
            gram = sparse_gram.toarray(order="F")
            del sparse_gram
            gram[np.diag_indices(m)] += self.params["regularization"]
            LOGGER.info("EASE: Cholesky factorization")
            factor, lower = linalg.cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
            if not np.shares_memory(factor, gram):
                raise MemoryError("Cholesky copied Gram; the two-buffer memory strategy is not available")
            del gram
            self.weights_ = np.empty((m, m), dtype=np.float64, order="F")
            positive_count, negative_count = 0, 0
            progress_every = max(1, (m + block_size - 1) // block_size // 10)
            for block_number, left in enumerate(range(0, m, block_size)):
                columns = np.arange(left, min(left + block_size, m))
                rhs = np.zeros((m, len(columns)), dtype=np.float64, order="F")
                rhs[columns, np.arange(len(columns))] = 1
                inverse_columns = linalg.cho_solve((factor, lower), rhs, overwrite_b=True, check_finite=False)
                diagonal = inverse_columns[columns, np.arange(len(columns))].copy()
                if not (np.isfinite(inverse_columns).all() and (diagonal > 0).all()):
                    raise FloatingPointError("Invalid inverse columns or nonpositive inverse diagonal")
                inverse_columns /= -diagonal
                inverse_columns[columns, np.arange(len(columns))] = 0
                self.weights_[:, columns] = inverse_columns
                positive_count += int(np.count_nonzero(inverse_columns > 0))
                negative_count += int(np.count_nonzero(inverse_columns < 0))
                if block_number % progress_every == 0:
                    LOGGER.info("EASE: solved %d/%d target columns", columns[-1] + 1, m)
            del factor, rhs, inverse_columns
            LOGGER.info("EASE: computing train objective in blocks")
            losses = ease_objective(self.interactions_, self.weights_, regularization=self.params["regularization"], block_size=block_size)
            self.training_summary_ = {
                "n_users": self.interactions_.shape[0], "n_history_items": m,
                "n_positive_interactions": self.interactions_.nnz,
                "zero_positive_items": int((self.interactions_.getnnz(axis=0) == 0).sum()),
                "weights_storage_bytes": self.weights_.nbytes,
                "positive_coefficients": positive_count, "negative_coefficients": negative_count,
                "zero_coefficients": m * m - positive_count - negative_count,
                "diagonal_zero": bool((np.diag(self.weights_) == 0).all()),
                "weights_sha256": _weights_sha256(self.weights_),
                "training_context_sha256": self.context_.fingerprint,
                "orientation": "B[source_item, target_item]", "negative_coefficients_kept": True,
                "factor_reused_gram_buffer": True, "full_inverse_materialized": False,
                "memory_estimate_at_training": memory_check, **losses,
            }
            self.training_compute_seconds_ = time.perf_counter() - start
            self._store_cache(self.cache_path_, cache_key)
        self.training_summary_ = {**self.training_summary_, "cache_key": cache_key, "cache_hit": self.cache_hit_,
                                  "original_training_compute_seconds": self.training_compute_seconds_,
                                  "memory_check_at_current_fit": memory_check}

    def _score(self, user_indices, item_indices, *, timestamp, features=None) -> np.ndarray:
        return np.asarray(self.interactions_[user_indices] @ self.weights_[:, item_indices])
