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
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
REPORT_DEFAULTS = {"inventory": "data_inventory.csv", "eligible_manifest": "eligible_manifest.csv",
                   "summary": "summary", "environment": "environment.json", "predictions": "predictions"}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def resolve_paths(cfg: dict) -> dict:
    """Fixed paths are relative to the project, never to the launch directory."""
    cfg["data"]["root"] = str(project_path(cfg["data"]["root"]))
    if cfg["data"].get("manifest"):
        cfg["data"]["manifest"] = str(project_path(cfg["data"]["manifest"]))
    runtime = cfg["runtime"]
    validate_cpu_config(runtime)
    offset = float(runtime.get("timestamp_utc_offset_hours", 8))
    if not math.isfinite(offset) or not -24 < offset < 24:
        raise ValueError("runtime.timestamp_utc_offset_hours must be finite and between -24 and 24.")
    runtime["output"] = str(project_path(runtime["output"]))
    runtime["log_dir"] = str(project_path(runtime.get("log_dir", "logs")))
    try:
        interval = float(runtime.get("log_interval_seconds", 300))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime.log_interval_seconds must be a finite nonnegative number.") from exc
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("runtime.log_interval_seconds must be a finite nonnegative number.")
    runtime["log_interval_seconds"] = interval
    if runtime.get("detector_execution", "block") not in {"block", "step"}:
        raise ValueError("runtime.detector_execution must be block or step.")
    if runtime.get("task_cache_format", "packed") not in {"packed", "npz"}:
        raise ValueError("runtime.task_cache_format must be packed or npz.")
    if not math.isfinite(float(runtime.get("task_cache_memory_mb", 2048))) or float(runtime.get("task_cache_memory_mb", 2048)) < 0:
        raise ValueError("runtime.task_cache_memory_mb must be finite and nonnegative.")
    if int(runtime.get("task_cache_checkpoint_pairs", 16)) < 1:
        raise ValueError("runtime.task_cache_checkpoint_pairs must be positive.")
    if not isinstance(runtime.get("keep_intermediates", False), bool):
        raise ValueError("runtime.keep_intermediates must be a boolean.")
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
    with open(project_path(path), encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return resolve_paths(cfg)


def resolve_config(cfg: dict, n_train: int, channels: int) -> dict:
    cfg = resolve_paths(copy.deepcopy(cfg))
    m = cfg["model"]
    if n_train < 65:
        raise ValueError("At least 65 normal training samples are required.")
    small = n_train - (256 if m["window"] == "auto" else m["window"]) < 8192
    if m["window"] == "auto":
        m["window"] = min(128 if small else 256, ((n_train - 33) // 32) * 32)
    if m["d_model"] == "auto":
        m["d_model"] = 64 if small else 96
    if m["window"] < 32 or m["window"] % 32 or m["window"] >= n_train:
        raise ValueError("window must be a multiple of 32 and smaller than training length.")
    if m["d_model"] % m["heads"]:
        raise ValueError("d_model must be divisible by heads.")
    for name, default in (("marginal_prototypes", 64 if small else 128),
                          ("joint_prototypes", 128 if small else 256)):
        if m[name] == "auto":
            m[name] = default
        m[name] = min(int(m[name]), max(1, n_train - m["window"]))
        if m[name] < 1:
            raise ValueError("Prototype capacity must be positive.")
    m["channels"] = channels
    s2 = cfg["stage2"]
    s2.setdefault("sources", ["relation", "marginal", "joint"])
    if s2["train_pairs"] == "auto":
        s2["train_pairs"] = 1024 if n_train >= 100000 else 512
    if not 0 < cfg["data"]["train_fraction"] < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0 < cfg["calibration"]["alpha"] < 1:
        raise ValueError("alpha must be in (0, 1).")
    if s2["variant"] not in {"full", "mlp", "single_gru", "no_alignment", "zero_alignment", "fixed_gate", "independent_gru"}:
        raise ValueError("Unknown detector variant.")
    for section, names in (("stage1", ("epochs", "batch_size", "stride", "patience")),
                            ("stage2", ("epochs", "batch_pairs", "block_length", "train_pairs", "validation_pairs", "patience")),
                            ("runtime", ("feature_batch_size", "threads"))):
        if any(cfg[section][n] < 1 for n in names):
            raise ValueError(f"{section}: counts and lengths must be positive.")
    if not 0 <= s2["fixed_retention"] <= 1 or not s2["validation_seeds"]:
        raise ValueError("Invalid fixed retention or validation seeds.")
    if set(s2.get("sources", [])) - {"relation", "marginal", "joint"}:
        raise ValueError("Unknown detector evidence source.")
    from .metrics import (LABEL_METRICS, SCORE_METRICS, REGRESSION_METRICS,
                          LABEL_METRIC_ORDER, SCORE_METRIC_ORDER)
    # Apply the same detection protocol even when resuming an older configuration.
    cfg["evaluation"]["label_metrics"] = list(LABEL_METRIC_ORDER)
    cfg["evaluation"]["score_metrics"] = [name for name in SCORE_METRIC_ORDER
        if not (cfg["evaluation"].get("skip_vus", False) and name.startswith("VUS_"))]
    cfg["evaluation"]["regression_metrics"] = []
    cfg["evaluation"]["supplementary_label_metrics"] = []
    for key, allowed in (("label_metrics", LABEL_METRICS), ("supplementary_label_metrics", LABEL_METRICS),
                         ("score_metrics", SCORE_METRICS), ("regression_metrics", REGRESSION_METRICS)):
        names = cfg["evaluation"].get(key, [])
        if not isinstance(names, (list, tuple)) or any(name not in allowed for name in names):
            raise ValueError("evaluation." + key + " must contain supported metric names.")
    seasonality = cfg["evaluation"].get("regression_seasonality", 1)
    if isinstance(seasonality, bool) or not isinstance(seasonality, int) or not 0 < seasonality < n_train:
        raise ValueError("evaluation.regression_seasonality must be a positive integer shorter than normal training history.")
    return cfg


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
    # Exact source aliases preserve checkpoints across reviewed equivalent implementations.
    compatibility = ROOT / "configs" / "implementation_compatibility.json"
    if compatibility.is_file():
        record = json.loads(compatibility.read_text(encoding="utf-8"))
        if record.get("format") != 1 or not isinstance(record.get("aliases"), dict):
            raise ValueError("Invalid implementation compatibility record.")
        return record["aliases"].get(identity, identity)
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
