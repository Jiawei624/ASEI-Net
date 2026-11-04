"""Use the inspected TAB source directly, without importing its baseline suite."""
from __future__ import annotations

import importlib
from functools import lru_cache
import subprocess
import sys
from pathlib import Path

from .config import ROOT, TAB_COMMIT


@lru_cache(maxsize=None)
def activate_tab(path: str | Path | None = None) -> Path:
    path = Path(path or ROOT / "third_party" / "TAB").resolve()
    if not (path / "ts_benchmark" / "data" / "utils.py").is_file():
        raise FileNotFoundError("TAB source missing. Run: python scripts/bootstrap_tab.py")
    try:
        revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Use the pinned TAB Git checkout; bootstrap_tab.py can create it.") from exc
    if revision != TAB_COMMIT:
        raise RuntimeError(f"Expected TAB {TAB_COMMIT}, found {revision}.")
    clean = subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--",
                            "ts_benchmark/data/utils.py", "ts_benchmark/evaluation/metrics"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if clean.returncode != 0:
        raise RuntimeError("TAB data/metric source differs from the pinned commit; use a clean checkout.")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    loaded = sys.modules.get("ts_benchmark")
    if loaded and getattr(loaded, "__file__", None):
        try:
            Path(loaded.__file__).resolve().relative_to(path)
        except ValueError as exc:
            raise RuntimeError("Another ts_benchmark package is already loaded.") from exc
    return path


def tab_module(name: str, path=None):
    activate_tab(path)
    return importlib.import_module("ts_benchmark." + name)
