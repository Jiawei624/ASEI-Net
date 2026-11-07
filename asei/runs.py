"""Timestamped experiment directories and explicit reuse of prior runs."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import time
import uuid

from .config import log_path, project_path, write_json


def timestamp(cfg):
    offset = float(cfg["runtime"].get("timestamp_utc_offset_hours", 8))
    return datetime.now(timezone(timedelta(hours=offset))).strftime("%Y%m%d_%H%M%S_%f")


def new_directory(parent, cfg):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    name = timestamp(cfg)
    while True:
        path = parent / name
        try:
            path.mkdir()
            return path
        except FileExistsError:
            name = timestamp(cfg) + "_" + uuid.uuid4().hex[:8]


def _runs_root(cfg):
    subdir = Path(cfg["runtime"].get("run_subdir", "runs"))
    if subdir.is_absolute() or not subdir.parts or ".." in subdir.parts:
        raise ValueError("runtime.run_subdir must be a relative directory within runtime.output.")
    return project_path(cfg["runtime"]["output"]) / subdir


def start_run(cfg, resume=None):
    runtime = cfg["runtime"]
    if resume is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", resume):
            raise ValueError("--resume must be a run ID printed by a previous training command.")
        directory = _runs_root(cfg) / resume
        path = directory / "run.json"
        if not path.is_file():
            raise FileNotFoundError("Run not found: " + str(directory))
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("format") != 1 or record.get("run_id") != resume:
            raise ValueError("Invalid run record: " + str(path))
    else:
        directory = new_directory(_runs_root(cfg), cfg)
        record = {"format": 1, "run_id": directory.name, "created_ns": time.time_ns(),
                  "timestamp_utc_offset_hours": runtime.get("timestamp_utc_offset_hours", 8)}
    runtime.update(run_id=record["run_id"], run_directory=str(directory),
                   run_created_ns=record["created_ns"])
    if resume is None:
        write_json(directory / "run.json", {**record, "config": cfg})
    return directory


def training_root(cfg, stage):
    run_id = cfg["runtime"].get("run_id")
    if run_id is None and stage not in {"all", "representation"}:
        raise ValueError("Continuing a stage requires --resume <run_id>.")
    return start_run(cfg, resume=run_id)


def run_log_path(cfg):
    path = log_path(cfg)
    return path.with_name(path.stem + "_" + cfg["runtime"]["run_id"] + (path.suffix or ".log"))
