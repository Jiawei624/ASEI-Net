"""One detection-metric CSV per command and bounded artifact cleanup."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

from .data import DATASETS
from .metrics import (LABEL_METRIC_ORDER, SCORE_METRIC_ORDER, EVENT_METRIC_ORDER, METRICS_PROTOCOL,
                      TAB_METRICS_PROTOCOL, CONTINUOUS_TOPK_PROTOCOLS)

METRIC_COLUMNS = LABEL_METRIC_ORDER + SCORE_METRIC_ORDER
CSV_COLUMNS = ("run_id", "run_created_ns", "dataset", "entity", "seed", "variant", "method", "metrics_protocol",
               "metrics_type", "test_points", "threshold_method", "anomaly_ratio", "alpha", "threshold") + METRIC_COLUMNS + EVENT_METRIC_ORDER + (
               "missing_metrics", "undefined_metrics", "model_file")


def result_row(record):
    row = {key: record.get(key, "") for key in CSV_COLUMNS[:7]}
    row["method"] = record.get("method") or record.get("config", {}).get("stage2", {}).get("method") or "in_1_2_2"
    row["metrics_protocol"] = record.get("metrics_protocol", record["metrics"].get("protocol", "legacy_binary_topk_v1"))
    row["test_points"] = record.get("test_points", "")
    row["metrics_type"] = record.get("metrics_type", "both")
    calibration = record.get("calibration", {})
    row["threshold_method"] = calibration.get("method", "normal_validation")
    row.update({key: calibration.get(key, "") for key in ("anomaly_ratio", "alpha", "threshold")})
    row["model_file"] = record.get("model_file", "")
    missing, undefined = [], []
    for name in METRIC_COLUMNS:
        relevant = (row["metrics_type"] == "both" or
                    (row["metrics_type"] == "label" and name in LABEL_METRIC_ORDER) or
                    (row["metrics_type"] == "score" and name in SCORE_METRIC_ORDER))
        if not relevant:
            row[name] = ""
            continue
        group = record["metrics"].get("label" if name in LABEL_METRIC_ORDER else "score", {})
        value = group.get(name)
        if name not in group:
            missing.append(name)
        elif value is None or not math.isfinite(float(value)):
            undefined.append(name)
        if name == "precision_at_k" and row["metrics_protocol"] not in CONTINUOUS_TOPK_PROTOCOLS:
            value = None
            if name not in missing and name not in undefined:
                undefined.append(name + ":legacy_binary_input")
        row[name] = value if value is not None and math.isfinite(float(value)) else ""
    events = record["metrics"].get("event", {}) if row["metrics_type"] != "score" else {}
    for name in EVENT_METRIC_ORDER:
        value = events.get(name)
        row[name] = value if value is not None and math.isfinite(float(value)) else ""
    row["missing_metrics"], row["undefined_metrics"] = ";".join(missing), ";".join(undefined)
    return row


def result_rows(record):
    if "label_by_ratio" not in record["metrics"]:
        return [result_row(record)]
    common = dict(record, metrics_protocol=record.get("metrics_protocol", TAB_METRICS_PROTOCOL))
    rows = []
    for entry in record["metrics"]["label_by_ratio"]:
        row = dict(common, metrics_type="label", metrics={"label": entry["metrics"], "event": entry.get("event", {})},
                   calibration={"method": "tab_anomaly_ratio", "anomaly_ratio": entry["anomaly_ratio"], "threshold": entry["threshold"]})
        rows.append(result_row(row))
    rows.append(result_row(dict(common, metrics_type="score", metrics_protocol=record["metrics"].get("score_protocol", METRICS_PROTOCOL),
                                metrics={"score": record["metrics"]["score"]},
                                calibration={"method": "tab_anomaly_ratio"})))
    return rows


def _model_identity(row):
    return tuple(str((row.get(k) or "in_1_2_2") if k == "method" else row.get(k, ""))
                 for k in ("run_id", "dataset", "entity", "seed", "variant", "method"))


def _identity(row):
    ratio = row.get("anomaly_ratio")
    ratio = float(ratio) if ratio not in (None, "") else ""
    return _model_identity(row) + (row.get("metrics_protocol", ""), row.get("metrics_type") or "both", ratio)


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    order = {name: i for i, name in enumerate(DATASETS)}
    rows = sorted(rows, key=lambda r: (order.get(r["dataset"], 99), str(r.get("entity", "")),
                                     int(r["seed"]), str(r.get("variant", "")), str(r.get("run_id", "")),
                                     r.get("metrics_type") or "both", float(r.get("anomaly_ratio") or 0)))
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({k: ((row.get(k) or "in_1_2_2") if k == "method" else
                             (row.get(k) or "both") if k == "metrics_type" else row.get(k, ""))
                          for k in CSV_COLUMNS} for row in rows)
    temporary.replace(path)
    return path


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_run_result(run_directory, record):
    path = Path(run_directory) / "metrics.csv"
    additions = result_rows(record)
    rows = read_rows(path) if path.exists() else []
    rows = [item for item in rows if _model_identity(item) != _model_identity(additions[0])] + additions
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
        if "label_by_ratio" in record["metrics"]:
            record = copy.deepcopy(record)
            for i, entry in enumerate(record["metrics"]["label_by_ratio"]):
                metrics, alarms = evaluate(values["labels"], values["scores"], values["valid"], entry["threshold"],
                    {"label_metrics": list(LABEL_METRIC_ORDER), "score_metrics": []})
                if "alarms_by_ratio" in values and not np.array_equal(alarms, values["alarms_by_ratio"][i]):
                    raise ValueError("Saved ratio alarms differ from scores/thresholds: " + str(predictions))
                entry["metrics"] = metrics["label"]
                entry["undefined_or_warnings"] = metrics["undefined_or_warnings"]
            return record
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
    csv_models = set()
    csv_paths = [root] if root.is_file() and root.suffix == ".csv" else sorted(root.rglob("*.csv"))
    for path in csv_paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            # Named per-dataset result files are valid too. Check the header
            # before reading rows so raw data and summary tables are skipped.
            if not {"dataset", "entity", "seed", "variant"}.issubset(reader.fieldnames or []):
                continue
            for row in reader:
                if row.get("dataset") in DATASETS:
                    row["method"] = row.get("method") or "in_1_2_2"
                    row["metrics_type"] = row.get("metrics_type") or "both"
                    row["threshold_method"] = row.get("threshold_method") or "normal_validation"
                    row["metrics_protocol"] = row.get("metrics_protocol") or "legacy_binary_topk_v1"
                    if row["metrics_protocol"] not in CONTINUOUS_TOPK_PROTOCOLS and row.get("precision_at_k"):
                        row["precision_at_k"] = ""
                        row["undefined_metrics"] = ";".join(filter(None, [row.get("undefined_metrics"), "precision_at_k:legacy_binary_input"]))
                    row["_source"] = str(path)
                    rows[_identity(row)] = row
                    csv_models.add(_model_identity(row))
    paths = [root] if root.is_file() and root.suffix == ".json" else sorted(root.rglob("*.json"))
    for path in paths:
        if "task_cache" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not {"dataset", "entity", "seed", "metrics", "variant"}.issubset(data):
            continue
        if data["dataset"] in DATASETS:
            if not recompute_labels and _model_identity(data) in csv_models:
                continue
            if recompute_labels:
                data = _refresh_labels(data, path)
                rows = {key: row for key, row in rows.items() if _model_identity(row) != _model_identity(data)}
            for row in result_rows(data):
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
             "resolved_config.json", "data_provenance.json", "normal_features.npz", "test_features.npz", "task_cache",
             "teacher.pt", "teacher_history.json", "teacher_train_features.npz", "teacher_selection_features.npz",
             "threshold_train_features.npz")]
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
