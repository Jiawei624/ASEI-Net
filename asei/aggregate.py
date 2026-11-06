from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json
from .data import DATASETS
from .results import collect_rows, METRIC_COLUMNS
from .metrics import LABEL_METRIC_ORDER


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
        row["seed"] = int(row["seed"])
        row["run_id"] = data.get("run_id", "legacy")
        row["run_created_ns"] = int(data.get("run_created_ns") or 0)
        row["metrics_protocol"] = data["metrics_protocol"]
        for name in METRIC_COLUMNS:
            prefix = "label." if name in LABEL_METRIC_ORDER else "score."
            value = data.get(name)
            row[prefix + name] = float(value) if value not in (None, "") else float("nan")
        row["path"] = data.get("_source", "")
        rows.append(row)
    if not rows:
        raise ValueError("No completed detection results found.")
    frame = pd.DataFrame(rows)
    keys = ["dataset", "entity", "seed", "variant", "metrics_protocol"]
    if frame.duplicated(keys + ["run_id"]).any():
        raise ValueError("Duplicate entity/seed/variant results within the same run; select one output root.")
    keep = []
    for _, group in frame.groupby(keys, sort=False):
        if len(group) > 1 and not (group.run_created_ns > 0).any():
            raise ValueError("Duplicate legacy entity/seed/variant results; select one output root.")
        keep.append(group.sort_values(["run_created_ns", "run_id", "path"]).index[-1])
    return frame.loc[keep].reset_index(drop=True)


def aggregate(root, output, seeds=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = collect(root, seeds)
    keys = ["dataset", "entity", "seed", "variant", "path", "run_id", "run_created_ns", "metrics_protocol"]
    metrics = [c for c in frame if c not in keys]
    # Keep seed-level rows and separate metric protocols; average entities equally.
    groups = frame.groupby(["dataset", "seed", "variant", "metrics_protocol"], sort=True)
    summary = groups[metrics].mean().reset_index()
    summary = summary.rename(columns={name: name.split(".", 1)[1] for name in metrics})
    summary["entities"] = groups.size().to_numpy()
    summary["run_ids"] = groups.run_id.agg(lambda values: ";".join(sorted(set(values)))).to_numpy()
    summary.to_csv(output / "dataset_metrics.csv", index=False, encoding="utf-8-sig")
    return output


def paired_comparison(root, baseline_root, metric, variant, baseline_variant, samples=10000, seed=0, seeds=None):
    left, right = collect(root, seeds), collect(baseline_root, seeds)
    left, right = left[left.variant == variant], right[right.variant == baseline_variant]
    if metric not in left or metric not in right:
        raise ValueError(f"Metric is absent: {metric}")
    joined = left.merge(right, on=["dataset", "entity", "seed"], suffixes=("_method", "_baseline"))
    if (joined.metrics_protocol_method != joined.metrics_protocol_baseline).any():
        raise ValueError("Paired comparisons require the same metrics protocol on both sides.")
    joined["difference"] = joined[metric + "_method"] - joined[metric + "_baseline"]
    joined = joined[np.isfinite(joined.difference)]
    if joined.empty:
        raise ValueError("No paired entity/seed measurements.")
    # Average seeds for each entity before resampling the entity sampling unit.
    entity = joined.groupby(["dataset", "entity"]).difference.mean().reset_index()
    groups = [group.difference.to_numpy() for _, group in entity.groupby("dataset")]
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(samples)
    for values in groups:
        bootstrap += rng.choice(values, (samples, len(values)), replace=True).mean(1) / len(groups)
    return {"metric": metric, "difference": float(np.mean([x.mean() for x in groups])),
            "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(), "paired_entity_seed_runs": len(joined),
            "paired_entities": len(entity), "datasets": len(groups), "resamples": samples,
            "unit": "entity, stratified by dataset after averaging paired seeds"}
