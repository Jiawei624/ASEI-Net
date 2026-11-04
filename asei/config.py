from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAB_COMMIT = "5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2"
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
    runtime["output"] = str(project_path(runtime["output"]))
    runtime["log_dir"] = str(project_path(runtime.get("log_dir", "logs")))
    reports = cfg.setdefault("reports", {})
    for name, value in REPORT_DEFAULTS.items():
        reports.setdefault(name, value)
    return cfg


def report_path(cfg: dict, name: str, output_root=None) -> Path:
    path = Path(cfg.get("reports", {}).get(name, REPORT_DEFAULTS[name])).expanduser()
    root = project_path(output_root if output_root is not None else cfg["runtime"]["output"])
    return (path if path.is_absolute() else root / path).resolve()


def log_path(cfg: dict) -> Path:
    path = Path(cfg["runtime"].get("log_file", "gpu1_main.log")).expanduser()
    return (path if path.is_absolute() else project_path(cfg["runtime"]["log_dir"]) / path).resolve()


def configure_cuda(cfg: dict):
    """Apply configured device visibility before CUDA is initialized."""
    runtime = cfg["runtime"]
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
    files = {p.name: hashlib.sha256(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()
             for p in sorted((ROOT / "asei").glob("*.py"))}
    return fingerprint({"source_files": files, "tab_commit": TAB_COMMIT})


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
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
