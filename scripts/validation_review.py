"""Prepare frozen controls and inspect saved results without changing ASEI training.

See scripts/VALIDATION_STEPS.md for the experiment sequence and interpretation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def absolute(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_profiles(round_name, datasets):
    import yaml
    from asei.config import implementation_fingerprint, load_config, write_json

    if not re.fullmatch(r"[A-Za-z0-9_]+", round_name):
        raise ValueError("Round names may contain only letters, digits and underscores.")
    datasets = [name.lower() for name in datasets]
    if not datasets or set(datasets) - {"psm", "pump", "smd", "swat"} or len(set(datasets)) != len(datasets):
        raise ValueError("Choose distinct PSM/PUMP/SMD/SWAT datasets.")
    destination = ROOT / "configs" / "validation" / ("round" + round_name)
    output = "outputs/validation/round" + round_name
    content, source_hashes = {}, {}
    for dataset in datasets:
        source = ROOT / "configs" / (dataset + ".yaml")
        dependency, seen = source, set()
        while dependency not in seen:
            seen.add(dependency)
            source_hashes[dependency.relative_to(ROOT).as_posix()] = sha256(dependency)
            parent = yaml.safe_load(dependency.read_text(encoding="utf-8")).get("base")
            if parent is None:
                break
            dependency = (dependency.parent / parent).resolve()
        # Materialize inheritance so a frozen profile never follows later edits
        # to the common method file or depends on its relative source location.
        cfg = copy.deepcopy(load_config(source))
        cfg["stage2"]["variant"] = "full"
        cfg["runtime"].update(output=output, keep_intermediates=True, save_predictions=True,
                              seeds=[42], log_dir="logs/validation/round" + round_name)
        content[destination / (dataset + ".yaml")] = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        control = dict(base=dataset + ".yaml", stage2=dict(variant="single_gru", lambda_state=0.))
        content[destination / (dataset + "_single_gru.yaml")] = yaml.safe_dump(control, sort_keys=False)
    # Validate every destination before any write. Existing snapshots are immutable.
    for path, text in content.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Snapshot differs; use a new round name: " + str(path))
    manifest = dict(format="asei-validation-profiles/1", round=round_name, datasets=datasets,
        baseline="current local dataset configurations; not a server-verified round012 reproduction",
        source_sha256=source_hashes, implementation_at_preparation=implementation_fingerprint(),
        output=output, profiles={path.relative_to(ROOT).as_posix(): hashlib.sha256(text.encode("utf-8")).hexdigest()
                                for path, text in content.items()})
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise FileExistsError("Manifest differs; use a new round name: " + str(manifest_path))
    destination.mkdir(parents=True, exist_ok=True)
    for path, text in content.items():
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return manifest


def longest_run(values):
    import numpy as np
    values = np.asarray(values, bool)
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(lengths.max()) if len(lengths) else 0


def event_details(labels, alarms, valid, window=128, confirmation=8):
    import numpy as np
    from asei.tasks import event_ranges
    labels, alarms, valid = np.asarray(labels, bool), np.asarray(alarms, bool), np.asarray(valid, bool)
    if labels.ndim != 1 or labels.shape != alarms.shape or labels.shape != valid.shape:
        raise ValueError("Events, alarms and validity must share the full time axis.")
    if confirmation < 1 or window < 1:
        raise ValueError("Window and confirmation must be positive.")
    events, rows = event_ranges(labels), []
    for number, (start, stop) in enumerate(events):
        length = stop - start
        local, hits = alarms[start:stop], np.flatnonzero(alarms[start:stop])
        limit = events[number + 1][0] if number + 1 < len(events) else len(labels)
        recovery = None
        if limit - stop >= confirmation:
            good = valid[stop:limit] & ~alarms[stop:limit]
            confirmed = np.flatnonzero(np.convolve(good.astype(np.int64), np.ones(confirmation, np.int64), "valid") == confirmation)
            recovery = int(confirmed[0]) if len(confirmed) else None
        rows.append(dict(event=number + 1, start=start, stop_exclusive=stop, length=length,
            duration_group="le_W" if length <= window else "W_to_4W" if length <= 4 * window else "gt_4W",
            detected=bool(len(hits)), detection_delay=int(hits[0]) if len(hits) else None,
            coverage=float(local.mean()), latter_half_coverage=float(local[length // 2:].mean()),
            onset8_coverage=float(local[:8].mean()),
            after_W_coverage=float(local[window:].mean()) if length > window else None,
            after_4W_coverage=float(local[4 * window:].mean()) if length > 4 * window else None,
            longest_missed_gap=longest_run(~local), longest_missed_fraction=longest_run(~local) / length,
            scorable_fraction=float(valid[start:stop].mean()), recovery_delay=recovery,
            recovery_censored=recovery is None,
            recovery_confirmed_at=stop + recovery + confirmation - 1 if recovery is not None else None))
    return rows


def summarize_arrays(result, calibration_record, arrays, with_vus=False):
    import numpy as np
    from sklearn.metrics import average_precision_score
    from asei.calibration import ParallelCalibration
    from asei.metrics import METRICS_PROTOCOL, _evaluation_scores, event_metrics, score_metrics
    from asei.tasks import event_ranges

    if result.get("metrics_protocol") != METRICS_PROTOCOL:
        raise ValueError("Unexpected evaluation protocol; keep incompatible protocols separate.")
    labels = np.asarray(arrays["labels"])
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected the original binary ground-truth time axis.")
    if len(labels) != result["test_points"]:
        raise ValueError("Prediction length differs from result.json.")
    raw = {}
    for key in ("short", "dynamic", "short_valid", "dynamic_valid"):
        value = np.asarray(arrays["raw_" + key])
        if value.shape != labels.shape:
            raise ValueError("Invalid prediction field: raw_" + key)
        raw[key] = value.astype(bool) if key.endswith("valid") else value
    calibration = ParallelCalibration.from_dict(calibration_record)
    output = calibration.transform(raw["short"], raw["dynamic"], raw["short_valid"], raw["dynamic_valid"])
    for key in ("scores", "p_mix", "p_short", "p_dynamic"):
        if key in arrays:
            np.testing.assert_allclose(output[key], arrays[key], rtol=1e-9, atol=1e-12, equal_nan=True,
                                       err_msg="Bundle calibration and saved predictions differ: " + key)
    if "valid" in arrays and not np.array_equal(output["valid"], arrays["valid"]):
        raise ValueError("Saved validity differs from bundle calibration.")
    normal, anomaly = labels == 0, labels == 1
    ranking, warnings = [], {}
    score_names = ["auc_roc", "auc_pr"] + (["VUS_ROC", "VUS_PR"] if with_vus else [])
    for name, score, valid in (("short_raw", raw["short"], raw["short_valid"]),
                               ("dynamic_raw", raw["dynamic"], raw["dynamic_valid"]),
                               ("fused_rank" if calibration.ranking == "empirical" else "fused_continuous_tail", output["scores"], output["valid"])):
        _, _, valid, full_scores = _evaluation_scores(labels, score, valid)
        metrics = score_metrics(labels, full_scores, score_names, warnings) if valid.any() else {}
        ap = float(average_precision_score(labels, full_scores)) if valid.any() and len(np.unique(labels)) == 2 else None
        row = dict(signal=name, scorable_fraction=float(valid.mean()), AP=ap, **metrics)
        if valid.any():
            row["unique_scores"] = len(np.unique(np.asarray(score)[valid]))
            row["tied_fraction"] = 1 - row["unique_scores"] / int(valid.sum())
        ranking.append(row)
    workpoints, events, channels = [], [], []
    alphas = np.asarray(arrays.get("alphas", result["config"]["calibration"]["alphas"]), float)
    window = int(result["config"]["model"]["window"])
    confirmation = int(result["config"]["evaluation"]["recovery_confirmation"])
    names = result["provenance"]["channels"]
    for a_index, alpha in enumerate(alphas):
        alarms = output["valid"] & (output["p_mix"] <= alpha)
        if "alarms_by_alpha" in arrays and not np.array_equal(alarms, arrays["alarms_by_alpha"][a_index]):
            raise ValueError("Saved alarms differ from bundle calibration.")
        tp, fp, fn = int((alarms & anomaly).sum()), int((alarms & normal).sum()), int((~alarms & anomaly).sum())
        point = dict(alpha=float(alpha), f_score=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn) if tp + fn else None,
            normal_fpr=float(alarms[normal].mean()) if normal.any() else None,
            false_alarm_events=len(event_ranges(alarms & normal)),
            **event_metrics(labels, alarms, output["valid"], confirmation))
        point["false_alarm_events_per_10000_normal_points"] = point["false_alarm_events"] * 10000 / int(normal.sum()) if normal.any() else None
        rows = event_details(labels, alarms, output["valid"], window, confirmation)
        point["longest_missed_gap_macro"] = float(np.mean([r["longest_missed_gap"] for r in rows])) if rows else None
        for row in rows:
            events.append(dict(alpha=float(alpha), **row))
        for branch, weight in zip(("short", "dynamic"), calibration.weights):
            branch_alarm = raw[branch + "_valid"] & (weight > 0) & (output["p_" + branch] <= alpha * weight)
            point[branch + "_normal_fpr"] = float(branch_alarm[normal].mean()) if normal.any() else None
            point[branch + "_anomaly_recall"] = float(branch_alarm[anomaly].mean()) if anomaly.any() else None
            attribution = arrays.get("raw_attribution_" + branch)
            if attribution is not None:
                counts = np.bincount(np.asarray(attribution)[normal & branch_alarm].astype(int), minlength=len(names))
                for channel, count in enumerate(counts):
                    channels.append(dict(alpha=float(alpha), branch=branch, channel=channel, name=names[channel],
                        normal_alarm_wins=int(count), fraction_of_branch_false_alarms=float(count / counts.sum()) if counts.sum() else 0.))
        workpoints.append(point)
    tails = {}
    for branch, reference in zip(("short", "dynamic"), calibration.references):
        valid = raw[branch + "_valid"]
        beyond = valid & (raw[branch] > reference[-1]) if len(reference) else np.zeros(len(labels), bool)
        tails[branch] = dict(reference_points=len(reference),
            normal_exceed_reference_max=float(beyond[normal & valid].mean()) if (normal & valid).any() else None,
            anomaly_exceed_reference_max=float(beyond[anomaly & valid].mean()) if (anomaly & valid).any() else None)
    summary = dict(format="asei-validation-diagnostics/1", dataset=result["dataset"], entity=result["entity"],
        seed=result["seed"], variant=result["variant"], run_id=result["run_id"], test_points=len(labels),
        purpose="retrospective diagnosis only; no model or threshold is selected from test labels",
        fractions_unit="0_to_1", AP_definition="sklearn average_precision_score; current auc_pr uses the same convention",
        ranking_mapping=calibration.ranking,
        ranking=ranking, tails=tails, workpoints=workpoints, warnings=warnings,
        not_measured=["forecast-error assimilation", "normal-reference error", "state/gate dependence"])
    coverage = arrays.get("raw_short_reference_coverage")
    if coverage is not None:
        summary["mean_short_reference_coverage"] = {
            name: float(np.asarray(coverage)[mask & raw["short_valid"]].mean()) if (mask & raw["short_valid"]).any() else None
            for name, mask in (("normal", normal), ("anomaly", anomaly))}
    return summary, ranking, events, channels


def diagnose(directory, with_vus=False):
    import numpy as np
    from asei.compat import load_torch_checkpoint
    from asei.config import write_json

    directory = absolute(directory)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    bundle = load_torch_checkpoint(directory / "bundle.pt", map_location="cpu")
    saved_result = bundle.get("experiment", {}).get("result", {})
    if saved_result and any(saved_result.get(key) != result.get(key) for key in ("run_id", "model_key", "variant", "dataset", "entity", "seed")):
        raise ValueError("Bundle and result.json belong to different experiments.")
    # Read only required scalar fields, even if a historical NPZ contains full state vectors.
    fields = ("labels", "alphas", "alarms_by_alpha", "scores", "p_mix", "p_short", "p_dynamic", "valid",
              "raw_short", "raw_dynamic", "raw_short_valid", "raw_dynamic_valid", "raw_attribution_short",
              "raw_attribution_dynamic", "raw_short_reference_coverage")
    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in fields if key in saved}
    summary, ranking, events, channels = summarize_arrays(result, bundle["calibration"], arrays, with_vus)
    summary["source_sha256"] = {name: sha256(directory / name) for name in ("result.json", "bundle.pt")}
    destination = directory / "review"
    write_json(destination / "diagnostics.json", summary)
    for name, rows in (("branch_scores.csv", ranking), ("event_details.csv", events), ("channel_false_alarms.csv", channels)):
        write_csv(destination / name, rows)
    return destination


def frozen_evidence_identity(directory, result):
    from asei.compat import load_torch_checkpoint
    from asei.config import fingerprint

    path = directory / "bundle.pt"
    if not path.exists():
        return dict(available=False, reason="bundle.pt missing")
    saved = load_torch_checkpoint(path, map_location="cpu")
    experiment, cfg = saved["experiment"], saved["config"]
    digest = hashlib.sha256()
    for name, tensor in sorted(saved["model"].items()):
        if name.startswith("normal."):
            array = tensor.detach().cpu().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode("utf-8"))
            digest.update(array.tobytes(order="C"))
    scope, recipes = experiment["calibration_scope"], {}
    channels, window = len(result["provenance"]["channels"]), cfg["model"]["window"]
    lengths = {"train": scope["fit_points"], "selection": scope["selection_points"] + (0 if scope["source"] == "independent_normal" else window)}
    for split, offset in (("train", 20000), ("selection", 30000)):
        key = fingerprint(dict(split=split, seed=result["seed"] + offset, stage2=cfg["stage2"],
            shape=(lengths[split], channels), normal_key=experiment["normal_key"], teacher_key=experiment["teacher_key"]))
        index_path = directory.parents[1] / "tasks" / key / split / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            recipes[split] = dict(tasks=len(index["entries"]), sha256=json_digest(index["entries"]),
                                  scope="recipes and labels only; numeric task payloads are not hashed")
        else:
            recipes[split] = dict(available=False, reason="retained task index not found")
    return dict(available=True, implementation_key=experiment["implementation_key"], normal_key=experiment["normal_key"],
        teacher_key=experiment["teacher_key"], frozen_normal_sha256=digest.hexdigest(), tasks=recipes)


def collect(root, output, include_predictions=False, run_ids=None):
    from asei.config import write_json

    root, output = absolute(root), absolute(output)
    results = sorted(root.rglob("result.json"))
    if not results:
        raise FileNotFoundError("No result.json under " + str(root))
    rows, groups, archive_files = [], {}, {}
    requested, found = set(run_ids or []), set()
    for path in results:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not {"metrics", "variant", "run_id", "provenance"}.issubset(result):
            continue
        if requested and result["run_id"] not in requested:
            continue
        found.add(result["run_id"])
        directory = path.parent
        common = {key: result[key] for key in ("run_id", "dataset", "entity", "seed", "variant", "metrics_protocol", "test_points")}
        selected = result.get("detector_selection", {})
        for point in result["metrics"]["label_by_alpha"]:
            row = {**common, "alpha": point["alpha"], **result["metrics"]["score"], **point["metrics"],
                   **point["event"], "normal_fpr": point["normal_fpr"], "selected_epoch": selected.get("best_epoch")}
            for name, branch in point.get("branches", {}).items():
                row.update({name + "_" + key: value for key, value in branch.items()})
            rows.append(row)
        identity = frozen_evidence_identity(directory, result)
        group = (result["run_id"], result["dataset"], result["entity"], result["seed"])
        groups.setdefault(group, []).append(dict(variant=result["variant"], **identity))
        wanted = [path, directory / "resolved_config.json", directory / "calibration.json"]
        wanted.extend((directory / "review").glob("*.json"))
        wanted.extend((directory / "review").glob("*.csv"))
        if include_predictions:
            wanted.append(directory / "predictions.npz")
        for source in wanted:
            if source.is_file():
                archive_files[source.relative_to(root).as_posix()] = source
        for ancestor in directory.parents:
            if ancestor == root.parent:
                break
            if (ancestor / "run.json").is_file():
                archive_files[(ancestor / "run.json").relative_to(root).as_posix()] = ancestor / "run.json"
                if (ancestor / "metrics.csv").is_file():
                    archive_files[(ancestor / "metrics.csv").relative_to(root).as_posix()] = ancestor / "metrics.csv"
                break
        log_dir = Path(result["config"]["runtime"].get("log_dir", "logs"))
        log_dir = log_dir if log_dir.is_absolute() else ROOT / log_dir
        for log in log_dir.glob("*_" + result["run_id"] + ".log"):
            archive_files["logs/" + log.name] = log
        from asei.runs import run_log_path
        runtime = result["config"]["runtime"]
        if "output" in runtime and "run_id" in runtime:
            named = absolute(runtime["output"]) / "results" / (run_log_path(result["config"]).stem + ".csv")
            if named.is_file():
                archive_files["results/" + named.name] = named
    if requested - found:
        raise ValueError("No completed results for run IDs: " + ", ".join(sorted(requested - found)))
    if not rows:
        raise ValueError("No compatible results were found.")
    checks = []
    for group, members in groups.items():
        comparison = dict(run_id=group[0], dataset=group[1], entity=group[2], seed=group[3], members=members)
        fields = ("implementation_key", "normal_key", "teacher_key", "frozen_normal_sha256")
        comparison["multiple_variants"] = len(members) > 1
        comparison["frozen_normal_equal"] = all(m.get("available") for m in members) and all(
            len({m.get(field) for m in members}) == 1 for field in fields)
        for split in ("train", "selection"):
            digests = [m.get("tasks", {}).get(split, {}).get("sha256") for m in members]
            comparison[split + "_recipes_equal"] = all(digests) and len(set(digests)) == 1
        checks.append(comparison)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary.csv", rows)
    write_json(output / "checks.json", dict(format="asei-validation-comparison/1", groups=checks,
        note="Single-member groups do not establish a controlled comparison. Recipe equality does not hash numerical task payloads."))
    archive_files["summary.csv"] = output / "summary.csv"
    archive_files["checks.json"] = output / "checks.json"
    archive = output / "feedback.zip"
    with zipfile.ZipFile(archive.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as stream:
        for name, source in sorted(archive_files.items()):
            stream.write(source, name)
    archive.with_suffix(".tmp").replace(archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create immutable control profiles from current dataset YAMLs")
    prepare.add_argument("--round", default="013")
    prepare.add_argument("--datasets", nargs="+", default=["PSM", "SMD", "PUMP", "SWAT"])
    diagnosis = sub.add_parser("diagnose", help="Read one detector's saved predictions and bundle calibration")
    diagnosis.add_argument("--detector-dir", required=True)
    diagnosis.add_argument("--with-vus", action="store_true", help="Also compute VUS for both raw branches; potentially expensive")
    collection = sub.add_parser("collect", help="Collect existing metrics and check identities; never select a model")
    collection.add_argument("--root", required=True)
    collection.add_argument("--output", required=True)
    collection.add_argument("--include-predictions", action="store_true")
    collection.add_argument("--run-ids", nargs="+", help="Include only these completed runs")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_profiles(args.round, args.datasets), ensure_ascii=False, indent=2))
    elif args.command == "diagnose":
        print(diagnose(args.detector_dir, args.with_vus))
    else:
        print(collect(args.root, args.output, args.include_predictions, args.run_ids))


if __name__ == "__main__":
    main()
