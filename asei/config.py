from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAB_COMMIT = "5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2"


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return cfg


def resolve_config(cfg: dict, n_train: int, channels: int) -> dict:
    cfg = copy.deepcopy(cfg)
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
