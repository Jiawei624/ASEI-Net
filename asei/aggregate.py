from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json
from .data import DATASETS
from .results import collect_rows, METRIC_COLUMNS
from .metrics import LABEL_METRIC_ORDER, EVENT_METRIC_ORDER


def flatten_numeric(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_numeric(value, name))
        elif value is None:
            result[name] = float("nan")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value if np.isfinite(value) else float("nan")
    return result


def collect(root, seeds=None):
    rows = []
    for data in collect_rows(root):
        if data["dataset"] not in DATASETS:
            continue
        if seeds is not None and int(data["seed"]) not in seeds:
            continue
        row = {k: data[k] for k in ("dataset", "entity", "seed", "variant")}
        row["method"] = data.get("method", "legacy")
        row["alpha"] = float(data["alpha"]) if data.get("alpha") not in (None, "") else float("nan")
        row["anomaly_ratio"] = float(data["anomaly_ratio"]) if data.get("anomaly_ratio") not in (None, "") else float("nan")
        row["metrics_type"] = data.get("metrics_type") or "both"
        row["threshold_method"] = data.get("threshold_method") or "normal_validation"
        row["seed"] = int(row["seed"])
        row["run_id"] = data.get("run_id", "legacy")
        row["run_created_ns"] = int(data.get("run_created_ns") or 0)
        row["metrics_protocol"] = data["metrics_protocol"]
        for name in METRIC_COLUMNS + EVENT_METRIC_ORDER:
            prefix = "event." if name in EVENT_METRIC_ORDER else "label." if name in LABEL_METRIC_ORDER else "score."
            value = data.get(name)
            row[prefix + name] = float(value) if value not in (None, "") else float("nan")
        for name in ("normal_fpr", "scorable_normal_fpr", "scorable_fraction", "short_calibration_points", "dynamic_calibration_points"):
            value = data.get(name)
            row["operating." + name] = float(value) if value not in (None, "") else float("nan")
        row["path"] = data.get("_source", "")
        rows.append(row)
    if not rows:
        raise ValueError("No completed detection results found.")
    frame = pd.DataFrame(rows)
    keys = ["dataset", "entity", "seed", "variant", "method", "metrics_protocol", "metrics_type", "threshold_method", "anomaly_ratio", "alpha"]
    if frame.duplicated(keys + ["run_id"]).any():
        raise ValueError("Duplicate entity/seed/variant results within the same run; select one output root.")
    keep = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        if len(group) > 1 and not (group.run_created_ns > 0).any():
            raise ValueError("Duplicate legacy entity/seed/variant results; select one output root.")
        keep.append(group.sort_values(["run_created_ns", "run_id", "path"]).index[-1])
    return frame.loc[keep].reset_index(drop=True)


def aggregate(root, output, seeds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = collect(root, seeds)
    keys = ["dataset", "entity", "seed", "variant", "method", "alpha", "anomaly_ratio", "metrics_type", "threshold_method", "path", "run_id", "run_created_ns", "metrics_protocol"]
    metrics = [c for c in frame if c not in keys]
    # Keep seed-level rows and separate metric protocols; average entities equally.
    groups = frame.groupby(["dataset", "seed", "variant", "method", "metrics_protocol", "metrics_type", "threshold_method", "anomaly_ratio", "alpha"], sort=True, dropna=False)
    summary = groups[metrics].mean().reset_index()
    summary = summary.rename(columns={name: name.split(".", 1)[1] for name in metrics})
    summary["entities"] = groups.size().to_numpy()
    summary["run_ids"] = groups.run_id.agg(lambda values: ";".join(sorted(set(values)))).to_numpy()
    summary.to_csv(output / "dataset_metrics.csv", index=False, encoding="utf-8-sig")
    return output


def paired_comparison(root, baseline_root, metric, variant, baseline_variant, samples=10000, seed=0, seeds=None, anomaly_ratio=None, alpha=None):
    left, right = collect(root, seeds), collect(baseline_root, seeds)
    left, right = left[left.variant == variant], right[right.variant == baseline_variant]
    label = metric.startswith(("label.", "event."))
    def select(frame):
        if "metrics_type" in frame:
            frame = frame[frame.metrics_type.isin(["label", "both"] if label else ["score", "both"])]
        if label and anomaly_ratio is not None and "anomaly_ratio" in frame:
            frame = frame[np.isclose(frame.anomaly_ratio, anomaly_ratio)]
        if label and alpha is not None and "alpha" in frame:
            frame = frame[np.isclose(frame.alpha, alpha)]
        return frame
    left, right = select(left), select(right)
    if metric not in left or metric not in right:
        raise ValueError(f"Metric is absent: {metric}")
    if left.duplicated(["dataset", "entity", "seed"]).any() or right.duplicated(["dataset", "entity", "seed"]).any():
        raise ValueError("Choose one anomaly ratio (--anomaly-ratio) or normal alpha (--alpha), method and metrics protocol per entity/seed on each side.")
    joined = left.merge(right, on=["dataset", "entity", "seed"], suffixes=("_method", "_baseline"))
    if (joined.metrics_protocol_method != joined.metrics_protocol_baseline).any():
        raise ValueError("Paired comparisons require the same metrics protocol on both sides.")
    if label and {"anomaly_ratio_method", "anomaly_ratio_baseline"}.issubset(joined) and joined.anomaly_ratio_method.notna().any():
        if not np.isfinite(joined[["anomaly_ratio_method", "anomaly_ratio_baseline"]]).all().all() or not np.allclose(joined.anomaly_ratio_method, joined.anomaly_ratio_baseline):
            raise ValueError("Paired label metrics require the same known anomaly ratio.")
    elif label and {"alpha_method", "alpha_baseline"}.issubset(joined):
        if not np.isfinite(joined[["alpha_method", "alpha_baseline"]]).all().all() or not np.allclose(joined.alpha_method, joined.alpha_baseline):
            raise ValueError("Paired alarm metrics require the same known calibration alpha.")
    if metric.startswith("event.recovery"):
        fields = ["event.recovery_confirmation_method", "event.recovery_confirmation_baseline"]
        if not np.isfinite(joined[fields]).all().all() or not np.allclose(joined[fields[0]], joined[fields[1]]):
            raise ValueError("Paired recovery metrics require the same confirmation length.")
    joined["difference"] = joined[metric + "_method"] - joined[metric + "_baseline"]
    joined = joined[np.isfinite(joined.difference)]
    if joined.empty:
        raise ValueError("No paired entity/seed measurements.")
    # The four TAB series are fixed entities. Resampling only entities after
    # averaging seeds produces a spurious zero-width interval for each dataset.
    groups = list(joined.groupby("dataset"))
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(samples)
    means, entity_count = [], 0
    for _, group in groups:
        entities = list(group.groupby("entity"))
        dataset_samples = np.zeros(samples)
        dataset_means = []
        for _, entity_runs in entities:
            values = entity_runs.difference.to_numpy()
            dataset_samples += rng.choice(values, (samples, len(values)), replace=True).mean(1) / len(entities)
            dataset_means.append(values.mean())
        bootstrap += dataset_samples / len(groups)
        means.append(np.mean(dataset_means))
        entity_count += len(entities)
    return {"metric": metric, "difference": float(np.mean(means)),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(), "paired_entity_seed_runs": len(joined),
            "paired_entities": entity_count, "datasets": len(groups), "resamples": samples,
            "unit": "paired training seed within each fixed entity; entity and dataset macro-average",
            "scope": "Training randomness on these fixed datasets, not uncertainty over new datasets."}
