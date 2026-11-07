from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import yaml

from .resources import validate_cpu_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "psm.yaml"
TAB_ANOMALY_RATIOS = (0.1, 0.5, 1., 2., 3., 5., 10., 15., 20., 25.)
REPORT_DEFAULTS = {"inventory": "data_inventory.csv", "eligible_manifest": "eligible_manifest.csv",
                   "summary": "summary", "environment": "environment.json", "predictions": "predictions"}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def resolve_paths(cfg: dict) -> dict:
    """Apply one model's defaults and resolve paths against the project root."""
    from .settings import defaults
    defaults(cfg)
    cfg["data"]["root"] = str(project_path(cfg["data"]["root"]))
    if cfg["data"].get("manifest"):
        cfg["data"]["manifest"] = str(project_path(cfg["data"]["manifest"]))
    runtime = cfg["runtime"]
    validate_cpu_config(runtime)
    offset = float(runtime.get("timestamp_utc_offset_hours", 8))
    if not math.isfinite(offset) or not -24 < offset < 24:
        raise ValueError("runtime.timestamp_utc_offset_hours must be between -24 and 24.")
    runtime["output"] = str(project_path(runtime["output"]))
    runtime["log_dir"] = str(project_path(runtime.get("log_dir", "logs")))
    interval = float(runtime.get("log_interval_seconds", 30))
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("runtime.log_interval_seconds must be finite and nonnegative.")
    runtime["log_interval_seconds"] = interval
    if runtime.get("detector_execution", "block") not in {"block", "step"}:
        raise ValueError("runtime.detector_execution must be block or step.")
    memory = float(runtime.get("task_cache_memory_mb", 256))
    if not math.isfinite(memory) or memory < 0:
        raise ValueError("runtime.task_cache_memory_mb must be finite and nonnegative.")
    if not isinstance(runtime.get("keep_intermediates", False), bool):
        raise ValueError("runtime.keep_intermediates must be boolean.")
    reports = cfg.setdefault("reports", {})
    for name, value in REPORT_DEFAULTS.items():
        reports.setdefault(name, value)
    return cfg


def report_path(cfg: dict, name: str, output_root=None) -> Path:
    path = Path(cfg.get("reports", {}).get(name, REPORT_DEFAULTS[name])).expanduser()
    root = project_path(output_root if output_root is not None else cfg["runtime"]["output"])
    return (path if path.is_absolute() else root / path).resolve()


def log_path(cfg: dict) -> Path:
    path = Path(cfg["runtime"].get("log_file", "main.log")).expanduser()
    return (path if path.is_absolute() else project_path(cfg["runtime"]["log_dir"]) / path).resolve()


def configure_cuda(cfg: dict, gpu=None, device=None):
    """Apply configured device visibility before CUDA is initialized."""
    runtime = cfg["runtime"]
    if gpu is not None:
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
            raise ValueError("--gpu must be a nonnegative physical GPU index.")
        if device is not None:
            raise ValueError("Choose --gpu or --device, not both.")
        runtime["cuda_visible_devices"] = str(gpu)
        runtime["device"] = "cuda:0"
    elif device is not None:
        runtime["device"] = device
    if str(runtime.get("device", "auto")) == "cpu":
        return
    settings = {}
    if runtime.get("cuda_device_order") is not None:
        settings["CUDA_DEVICE_ORDER"] = str(runtime["cuda_device_order"])
    if runtime.get("cuda_visible_devices") is not None:
        settings["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_visible_devices"])
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        if any(os.environ.get(key) != value for key, value in settings.items()):
            raise RuntimeError("CUDA is already initialized; set GPU visibility in config before starting this process.")
    os.environ.update(settings)


def load_config(path: str | Path) -> dict:
    """Load YAML/JSON, with optional relative base configuration inheritance."""
    def merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base

    def read(path, parents):
        path = path.resolve()
        if path in parents:
            raise ValueError("Cyclic base configuration: " + str(path))
        with path.open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        if not isinstance(cfg, dict):
            raise ValueError("Configuration must be a YAML mapping.")
        base = cfg.pop("base", None)
        return merge(read(path.parent / base, parents | {path}), cfg) if base else cfg

    return resolve_paths(read(project_path(path), set()))


def resolve_config(cfg: dict, n_train: int, channels: int) -> dict:
    from .settings import resolve
    return resolve(resolve_paths(copy.deepcopy(cfg)), n_train, channels)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      default=str).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_fingerprint() -> str:
    # Normalize newlines so the same checkout has the same identity on Windows/Linux.
    files = {p.relative_to(ROOT / "asei").as_posix():
             hashlib.sha256(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()
             for p in sorted((ROOT / "asei").rglob("*.py"))}
    identity = fingerprint({"source_files": files})
    return identity


def write_json(path: str | Path, value):
    """Write standards-compliant JSON; undefined metrics are represented by null."""
    import math
    import numpy as np

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.ndarray):
            return clean(v.tolist())
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, Path):
            return str(v)
        return v

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)
