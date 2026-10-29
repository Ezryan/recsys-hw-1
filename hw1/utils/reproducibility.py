"""Configuration, random seeds, thread limits, and an environment manifest.

Call configure_reproducibility before numerical work. PYTHONHASHSEED must be
set before Python/Jupyter starts; changing it inside a running kernel is too late.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THREAD_LIMITER: Any = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the shared JSON config; relative paths are rooted at the project."""
    config_path = Path(path) if path is not None else Path("configs/experiment.json")
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    seed = config["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("cpu_threads", "torch_interop_threads"):
        value = config["runtime"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"runtime.{key} must be a positive integer")
    return config


def resolve_data_dir(config: dict[str, Any]) -> Path:
    """Resolve RECSYS_DATA_DIR without downloading data or creating folders."""
    storage = config["storage"]
    path = Path(os.environ.get(storage["data_dir_env"]) or storage["data_dir_default"])
    path = path.expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def configure_reproducibility(config: dict[str, Any]) -> dict[str, Any]:
    """Seed global RNGs and return explicit NumPy/PyTorch generators.

    Callers must use the returned generators or explicitly seeded new ones;
    np.random.seed does not seed separately created default_rng instances.
    """
    runtime = config["runtime"]
    seed = config["seed"]
    threads = runtime["cpu_threads"]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = runtime["cublas_workspace_config"]

    # Lazy imports allow environment variables to take effect on a fresh process.
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    global _THREAD_LIMITER
    _THREAD_LIMITER = threadpool_limits(limits=threads)
    torch.set_num_threads(threads)
    interop_threads = runtime["torch_interop_threads"]
    if torch.get_num_interop_threads() != interop_threads:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "Configure thread limits before PyTorch work; restart the kernel."
            ) from error

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = runtime["deterministic_algorithms"]
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested in config but is unavailable.")

    return {
        "numpy": np.random.default_rng(seed),
        "torch": torch.Generator(device="cpu").manual_seed(seed),
    }


def collect_environment(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect the active environment; no seed or package settings are changed."""
    import psutil
    import torch
    from threadpoolctl import threadpool_info

    cpu_model = platform.processor()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    memory = psutil.virtual_memory()
    gpu_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append({
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            })
    packages = sorted(
        {f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions()}
    )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "cpu": {
            "model": cpu_model,
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "memory": {"total_bytes": memory.total, "available_bytes": memory.available},
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "devices": gpu_devices,
        },
        "reproducibility": {
            "seed": config["seed"],
            "python_hash_seed_env": os.environ.get("PYTHONHASHSEED"),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "thread_pools": threadpool_info(),
        },
        "packages": packages,
        "experiment_config": config,
    }


def save_environment(
    config: dict[str, Any], output: str | Path | None = None
) -> Path:
    """Write a local manifest; default location is excluded from Git."""
    output_path = Path(output or config["runtime"]["environment_report"])
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = collect_environment(config)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_reproducibility(config)
    print(f"Environment manifest: {save_environment(config, args.output)}")


if __name__ == "__main__":
    main()
