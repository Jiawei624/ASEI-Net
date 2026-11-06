"""One detection-metric CSV per command and bounded artifact cleanup."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

from .data import DATASETS
from .metrics import LABEL_METRIC_ORDER, SCORE_METRIC_ORDER, METRICS_PROTOCOL

METRIC_COLUMNS = LABEL_METRIC_ORDER + SCORE_METRIC_ORDER
CSV_COLUMNS = ("run_id", "run_created_ns", "dataset", "entity", "seed", "variant", "metrics_protocol",
               "test_points", "alpha", "threshold") + METRIC_COLUMNS + (
               "missing_metrics", "undefined_metrics", "model_file")


def result_row(record):
    row = {key: record.get(key, "") for key in CSV_COLUMNS[:6]}
    row["metrics_protocol"] = record.get("metrics_protocol", record["metrics"].get("protocol", "legacy_binary_topk_v1"))
    row["test_points"] = record.get("test_points", "")
    row.update({key: record.get("calibration", {}).get(key, "") for key in ("alpha", "threshold")})
    row["model_file"] = record.get("model_file", "")
    missing, undefined = [], []
    for name in METRIC_COLUMNS:
        group = record["metrics"].get("label" if name in LABEL_METRIC_ORDER else "score", {})
        value = group.get(name)
        if name not in group:
            missing.append(name)
        elif value is None or not math.isfinite(float(value)):
            undefined.append(name)
        if name == "precision_at_k" and row["metrics_protocol"] != METRICS_PROTOCOL:
            value = None
            if name not in missing and name not in undefined:
                undefined.append(name + ":legacy_binary_input")
        row[name] = value if value is not None and math.isfinite(float(value)) else ""
    row["missing_metrics"], row["undefined_metrics"] = ";".join(missing), ";".join(undefined)
    return row


def _identity(row):
    return tuple(str(row.get(k, "")) for k in ("run_id", "dataset", "entity", "seed", "variant"))


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    order = {name: i for i, name in enumerate(DATASETS)}
    rows = sorted(rows, key=lambda r: (order.get(r["dataset"], 99), str(r.get("entity", "")),
                                     int(r["seed"]), str(r.get("variant", "")), str(r.get("run_id", ""))))
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in CSV_COLUMNS} for row in rows)
    temporary.replace(path)
    return path


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_run_result(run_directory, record):
    path = Path(run_directory) / "metrics.csv"
    row = result_row(record)
    rows = read_rows(path) if path.exists() else []
    rows = [item for item in rows if _identity(item) != _identity(row)] + [row]
    return _write_rows(path, rows)


def remove_run_result(run_directory, dataset, entity, seed, variant):
    path = Path(run_directory) / "metrics.csv"
    if path.exists():
        rows = [row for row in read_rows(path) if
                (row["dataset"], row["entity"], row["seed"], row["variant"]) != (dataset, entity, str(seed), variant)]
        _write_rows(path, rows)


def _refresh_labels(record, path):
    import copy
    import numpy as np
    from .metrics import evaluate
    predictions = path.parent / "predictions.npz"
    if not predictions.is_file():
        raise FileNotFoundError("Recomputing label metrics needs the original predictions.npz beside " + str(path))
    with np.load(predictions, allow_pickle=False) as values:
        metrics, alarms = evaluate(values["labels"], values["scores"], values["valid"],
            record["calibration"]["threshold"], {"label_metrics": list(LABEL_METRIC_ORDER), "score_metrics": []})
        if "alarms" in values and not np.array_equal(alarms, values["alarms"]):
            raise ValueError("Saved alarms do not match the saved threshold/scores: " + str(predictions))
    record = copy.deepcopy(record)
    record["metrics"]["label"] = metrics["label"]
    record["metrics"]["protocol"] = METRICS_PROTOCOL
    record["metrics_protocol"] = METRICS_PROTOCOL
    return record


def collect_rows(root, recompute_labels=False):
    root = Path(root)
    rows = {}
    csv_paths = [root] if root.is_file() and root.suffix == ".csv" else sorted(root.rglob("metrics.csv"))
    for path in csv_paths:
        for row in read_rows(path):
            if row.get("dataset") in DATASETS:
                row["_source"] = str(path)
                rows[_identity(row)] = row
    paths = [root] if root.is_file() and root.suffix == ".json" else sorted(root.rglob("*.json"))
    for path in paths:
        if "task_cache" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not {"dataset", "entity", "seed", "metrics", "variant"}.issubset(data):
            continue
        if data["dataset"] in DATASETS:
            if recompute_labels:
                data = _refresh_labels(data, path)
            row = result_row(data)
            row["_source"] = str(path)
            if recompute_labels:
                rows[_identity(row)] = row
            else:
                rows.setdefault(_identity(row), row)
    return list(rows.values())


def export_results(root, output, recompute_labels=False):
    rows = collect_rows(root, recompute_labels=recompute_labels)
    if not rows:
        raise ValueError("No completed detection results found.")
    return _write_rows(output, rows)


def prune_intermediates(directory, detector_directory):
    """Delete only known training artifacts inside the selected entity directory."""
    base, detector = Path(directory).resolve(), Path(detector_directory).resolve()
    detector.relative_to(base)
    names = [base / name for name in ("representation.pt", "stage1_best.pt", "stage1_history.json",
             "resolved_config.json", "data_provenance.json", "normal_features.npz", "test_features.npz", "task_cache")]
    names += [detector / name for name in ("detector.pt", "stage2_history.json", "resolved_config.json",
              "calibration.json", "normal_validation_scores.npz", "predictions.npz", "result.json")]
    for path in names:
        resolved = path.resolve()
        resolved.relative_to(base)
        if resolved == base or path.is_symlink():
            raise ValueError("Refusing cleanup outside the experiment's generated artifacts.")
        if path.is_dir():
            for item in path.rglob("*"):
                if item.is_symlink():
                    raise ValueError("Refusing cleanup of a task cache containing symbolic links.")
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
