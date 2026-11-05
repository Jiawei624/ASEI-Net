from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json
from .data import DATASETS


def flatten_numeric(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            result[name] = value
    return result


def collect(root, seeds=None):
    rows = []
    for path in Path(root).rglob("result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not {"dataset", "entity", "seed", "metrics", "variant"}.issubset(data):
            continue
        if data["dataset"] not in DATASETS:
            continue
        if seeds is not None and data["seed"] not in seeds:
            continue
        row = {k: data[k] for k in ("dataset", "entity", "seed", "variant")}
        row["run_id"] = data.get("run_id", "legacy")
        row["run_created_ns"] = int(data.get("run_created_ns", 0))
        row.update(flatten_numeric(data["metrics"]))
        row["path"] = str(path)
        rows.append(row)
    if not rows:
        raise ValueError("No completed result.json files found.")
    frame = pd.DataFrame(rows)
    keys = ["dataset", "entity", "seed", "variant"]
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
    keys = ["dataset", "entity", "seed", "variant", "path", "run_id", "run_created_ns"]
    metrics = [c for c in frame if c not in keys]
    frame.to_csv(output / "entity_seed_results.csv", index=False)
    # Each entity contributes once per dataset/seed. Datasets contribute equally across datasets.
    dataset_seed = frame.groupby(["variant", "dataset", "seed"])[metrics].mean()
    dataset_seed.to_csv(output / "dataset_seed_macro.csv")
    summary = dataset_seed.groupby(["variant", "dataset"]).agg(["mean", "std", "count"])
    summary.to_csv(output / "dataset_summary.csv")
    cross = dataset_seed.groupby(["variant", "seed"])[metrics].mean()
    cross.to_csv(output / "cross_dataset_seed_macro.csv")
    cross.groupby("variant").agg(["mean", "std", "count"]).to_csv(output / "cross_dataset_summary.csv")
    counts = frame.groupby(["variant", "dataset", "seed"]).size().rename("entities").reset_index()
    counts.to_csv(output / "coverage.csv", index=False)
    write_json(output / "aggregation.json", {"source": str(Path(root).resolve()), "runs": len(frame),
        "datasets": sorted(frame.dataset.unique().tolist()), "seeds": sorted(frame.seed.unique().tolist()),
        "requested_seeds": list(seeds) if seeds is not None else None,
        "selected_runs": sorted(frame.run_id.unique().tolist()),
        "repeat_policy": "Latest completed result per dataset/entity/seed/variant; repeated runs are not additional seeds.",
        "policy": "entity macro within dataset, dataset macro across datasets, sample standard deviation over seeds; undefined values are excluded with counts, never imputed"})
    return output


def paired_comparison(root, baseline_root, metric, variant, baseline_variant, samples=10000, seed=0, seeds=None):
    left, right = collect(root, seeds), collect(baseline_root, seeds)
    left, right = left[left.variant == variant], right[right.variant == baseline_variant]
    if metric not in left or metric not in right:
        raise ValueError(f"Metric is absent: {metric}")
    joined = left.merge(right, on=["dataset", "entity", "seed"], suffixes=("_method", "_baseline"))
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
