from __future__ import annotations

import copy
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from .config import (configure_cuda, file_sha256, fingerprint, implementation_fingerprint,
                     resolve_config, resolve_paths, write_json)
from .compat import load_torch_checkpoint
from .data import Entity, Record, Standardizer, load_entity, validation_context
from .features import extract_features, FeatureSequence, score_features, StreamingDetector
from .metrics import evaluate, threshold_from_normal, point_adjusted_predictions, METRICS_PROTOCOL
from .model import NormalRepresentation, DualTrajectoryDetector
from .training import build_task_cache, fit_representation, fit_detector, seed_everything
from .resources import configure_cpu
from .runs import training_root
from .results import write_run_result, remove_run_result, prune_intermediates


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this PyTorch environment.")
        count = torch.cuda.device_count()
        if device.index is not None and not 0 <= device.index < count:
            raise ValueError(f"Requested {device}, but only {count} CUDA device(s) are visible. "
                             "Use logical indices after CUDA_VISIBLE_DEVICES is applied.")
        # Memory-stat APIs in older PyTorch releases do not initialize the allocator.
        torch.cuda.init()
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device


def _load_checkpoint(path, expected_key, device):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing prerequisite checkpoint: {path}")
    checkpoint = load_torch_checkpoint(path, map_location=device)
    if checkpoint["key"] != expected_key:
        raise ValueError(f"Checkpoint configuration/data differ: {path}. Use a new output directory.")
    return checkpoint


def _cached_features(record, model, cfg, path, key):
    if path.exists():
        return FeatureSequence.load(path, key), 0.0
    start = time.perf_counter()
    sequence = extract_features(record, model, cfg["runtime"]["feature_batch_size"],
                                log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 300),
                                progress_label=path.stem + " batches")
    if cfg["runtime"].get("keep_intermediates", False):
        sequence.save(path, key)
    return sequence, time.perf_counter() - start


def _normal_calibration_record(train, valid, window, independent):
    if independent:
        return valid, window
    # Reconstruct states using the final frozen model over the actual normal past.
    # Only the held-out suffix contributes scores to the threshold.
    record = Record(np.concatenate([train.z, valid.z[window:]]),
                     np.concatenate([train.mask, valid.mask[window:]]))
    return record, len(train.z)


def _latency(model, detector, scaler, threshold, record, points):
    w = model.cfg["window"]
    n = min(points, max(0, len(record.z) - w))
    if n == 0:
        return {"samples": 0, "p50_ms": None, "p95_ms": None}
    raw = record.z[:w + n].astype(np.float64) * scaler.scale + scaler.mean
    raw[~record.mask[:w + n]] = np.nan
    stream = StreamingDetector(model, detector, scaler, threshold)
    timings = []
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    for i, sample in enumerate(raw):
        if cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        stream.push(sample)
        if cuda:
            torch.cuda.synchronize(device)
        if i >= w:
            timings.append(1000 * (time.perf_counter() - start))
    return {"samples": n, "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)), "scope": "raw-sample preprocessing through final alarm, sequential batch=1"}


def save_bundle(path, representation, detector, scaler, calibration, cfg, provenance, experiment=None):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"format": 1, "model_config": cfg["model"], "representation": representation.state_dict(),
                "detector": detector.state_dict(),
                "detector_config": {k: cfg["stage2"].get(k) for k in ("variant", "fixed_retention", "sources")},
                "scaler": scaler.as_dict(), "threshold": calibration["threshold"],
                "channel_names": provenance["channels"], "experiment": experiment}, temporary)
    temporary.replace(path)


def load_bundle(path, device="cpu"):
    device = get_device(device)
    checkpoint = load_torch_checkpoint(path, map_location=device)
    if checkpoint.get("format") != 1:
        raise ValueError("Unsupported deployment bundle format.")
    model = NormalRepresentation(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["representation"])
    dc = checkpoint["detector_config"]
    detector = DualTrajectoryDetector(model.history.identity.weight, **dc).to(device)
    detector.load_state_dict(checkpoint["detector"])
    scaler = Standardizer(np.array(checkpoint["scaler"]["mean"], np.float64),
                          np.array(checkpoint["scaler"]["scale"], np.float64))
    for module in (model, detector):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return StreamingDetector(model, detector, scaler, checkpoint["threshold"]), checkpoint["channel_names"]


def _variant_name(cfg):
    name = cfg["stage2"]["variant"]
    if cfg["stage2"].get("sources") != ["relation", "marginal", "joint"]:
        name += "_" + ("_".join(cfg["stage2"].get("sources", [])) or "self")
    return name


def _calibrate_model(entity, train, valid, representation, detector, cfg, directory, representation_key, cal_key):
    normal_record, validation_start = _normal_calibration_record(
        train, valid, cfg["model"]["window"], entity.validation_path is not None)
    normal_features, _ = _cached_features(normal_record, representation, cfg,
        directory / "normal_features.npz", representation_key + "normal")
    output = score_features(normal_features, representation, detector, cfg["stage2"]["block_length"],
        log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 300),
        progress_label="calibration scoring blocks", execution=cfg["runtime"].get("detector_execution", "block"))
    normal_scores = output["scores"][validation_start:]
    calibration = {**threshold_from_normal(normal_scores, cfg["calibration"]["alpha"]), "key": cal_key,
                   "validation_targets": len(normal_scores), "scorable_fraction": float(np.isfinite(normal_scores).mean())}
    return calibration, normal_scores


def _evaluate_and_save(entity, train, valid, raw_test, labels, scaler, provenance, cfg, seed,
                       representation, detector, directory, detector_dir, calibration, experiment, diagnostics):
    if entity.test_continuous:
        normal_record, _ = _normal_calibration_record(train, valid, cfg["model"]["window"], entity.validation_path is not None)
        past = train if entity.validation_path is not None else normal_record
        test = Record(np.concatenate([past.z, raw_test.z]), np.concatenate([past.mask, raw_test.mask]))
        test_start = len(past.z)
    else:
        test, test_start = raw_test, 0
    features, feature_seconds = _cached_features(test, representation, cfg, directory / "test_features.npz",
                                                 experiment["representation_key"] + "test")
    began = time.perf_counter()
    output = score_features(features, representation, detector, cfg["stage2"]["block_length"], diagnostics,
        log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 300), progress_label="test scoring blocks",
        execution=cfg["runtime"].get("detector_execution", "block"))
    scores, ready = output["scores"][test_start:], output["valid"][test_start:]
    inference_seconds = time.perf_counter() - began
    print(f"evaluation label_metrics={len(cfg['evaluation']['label_metrics'])} "
          f"score_metrics={len(cfg['evaluation']['score_metrics'])}", flush=True)
    results, alarms = evaluate(labels, scores, ready, calibration["threshold"], cfg["evaluation"])
    histories = {}
    for name, file in (("representation", directory / "stage1_history.json"),
                       ("detector", detector_dir / "stage2_history.json"),
                       ("cache", directory / "task_cache/manifest.json")):
        if file.exists():
            histories[name] = json.loads(file.read_text(encoding="utf-8"))
    old_timing = (experiment.get("result") or {}).get("timing", {})
    device = next(representation.parameters()).device
    timings = {"stage1_seconds": histories.get("representation", {}).get("seconds", old_timing.get("stage1_seconds")),
               "stage2_seconds": histories.get("detector", {}).get("seconds", old_timing.get("stage2_seconds")),
               "cache_preparation_seconds": histories.get("cache", {}).get("preparation_seconds_total", old_timing.get("cache_preparation_seconds")),
               "cache_bytes": histories.get("cache", {}).get("bytes", old_timing.get("cache_bytes")),
               "test_feature_seconds_this_call": feature_seconds, "detector_inference_seconds": inference_seconds,
               "peak_cuda_bytes_this_call": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    measured = [timings[k] for k in ("stage1_seconds", "stage2_seconds", "cache_preparation_seconds")]
    timings["total_training_seconds"] = sum(measured) if all(v is not None for v in measured) else None
    keep = cfg["runtime"].get("keep_intermediates", False)
    if keep and cfg["evaluation"].get("latency_points", 0) > 0:
        timings["latency"] = _latency(representation, detector, scaler, calibration["threshold"], raw_test,
                                     cfg["evaluation"]["latency_points"])
    run_directory = Path(cfg["runtime"]["run_directory"])
    result = {"dataset": entity.dataset, "entity": entity.name, "seed": seed,
        "run_id": cfg["runtime"]["run_id"], "run_created_ns": cfg["runtime"]["run_created_ns"],
        "variant": _variant_name(cfg), "model_key": experiment["detector_key"],
        "implementation_key": experiment["implementation_key"], "metrics_protocol": METRICS_PROTOCOL,
        "test_points": len(labels), "calibration": calibration, "metrics": results, "timing": timings,
        "model_file": (detector_dir / "bundle.pt").relative_to(run_directory).as_posix(),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device), "cuda": torch.version.cuda,
            "packages": experiment["packages"]}, "config": cfg}
    experiment.update(result=result, calibration=calibration, calibration_config=copy.deepcopy(cfg["calibration"]),
                      execution=cfg["runtime"].get("detector_execution", "block"))
    # Histories are small metadata inside the selected model, never extra final files.
    summaries = experiment.setdefault("training_history", {})
    for name in ("representation", "detector"):
        if name in histories:
            summaries[name] = {k: histories[name][k] for k in ("epochs", "seconds") if k in histories[name]}
    save_bundle(detector_dir / "bundle.pt", representation, detector, scaler, calibration, cfg, provenance, experiment)
    csv_path = write_run_result(run_directory, result)
    if keep:
        prediction_data = {"scores": scores, "valid": ready, "alarms": alarms, "labels": labels,
                           "point_adjusted_alarms": point_adjusted_predictions(labels, alarms)}
        if diagnostics:
            prediction_data.update({k: output[k][test_start:] for k in
                ("retention", "state_difference", "marginal_difference", "joint_difference")})
        np.savez_compressed(detector_dir / "predictions.npz", **prediction_data)
        write_json(detector_dir / "result.json", result)
    else:
        prune_intermediates(directory, detector_dir)
    print(f"saved metrics {csv_path}; best model {detector_dir / 'bundle.pt'}", flush=True)
    return detector_dir


def run_entity(entity: Entity, config: dict, seed: int, stage="all", diagnostics=False):
    config = resolve_paths(copy.deepcopy(config))
    configure_cuda(config)
    configure_cpu(config)
    seed_everything(seed)
    device = get_device(config["runtime"]["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train, raw_valid, raw_test, labels, scaler, provenance = load_entity(entity, config["data"]["train_fraction"])
    if config["data"].get("multivariate_only", True) and train.z.shape[1] < 2:
        raise ValueError("Multivariate experiments require at least two observed training channels.")
    cfg = resolve_config(config, len(train.z), train.z.shape[1])
    w = cfg["model"]["window"]
    valid = validation_context(train, raw_valid, w, independent=entity.validation_path is not None)
    directory = training_root(cfg, stage) / entity.dataset / entity.name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    code_key = implementation_fingerprint()
    runtime_packages = {name: metadata.version(name) for name in
                        ("numpy", "pandas", "torch", "scikit-learn", "scipy", "statsmodels", "PyYAML")}
    runtime_identity = {"packages": runtime_packages, "python": list(sys.version_info[:2]),
                        "torch_build": str(torch.__version__)}
    stage1_identity = dict(cfg["stage1"], workers=cfg["runtime"]["effective_cpu"]["workers"])
    normal_key = fingerprint({"implementation": code_key, "model": cfg["model"], "stage1": stage1_identity,
                              "data": {k: v for k, v in provenance.items() if k != "source"},
                              "runtime": runtime_identity, "seed": seed})
    representation_path = directory / "representation.pt"
    checkpoint = _load_checkpoint(representation_path, normal_key, device) if representation_path.exists() else None
    variant_name = _variant_name(cfg)
    detector_dir = directory / "detectors" / variant_name
    selected_bundle = None
    representation_key = None
    if checkpoint is None:
        # A completed compact experiment stores both best networks in one bundle.
        paths = sorted(directory.glob("detectors/*/bundle.pt"),
                       key=lambda p: p.parent != detector_dir)
        for path in paths:
            saved = load_torch_checkpoint(path, map_location=device)
            identity = saved.get("experiment") or {}
            if identity.get("normal_key") != normal_key:
                continue
            checkpoint = {"key": normal_key, "model": saved["representation"]}
            representation_key = identity["representation_key"]
            if path.parent == detector_dir:
                selected_bundle = saved
            break
        if paths and checkpoint is None:
            raise ValueError("Saved model configuration/data differ; start a new run instead of replacing this experiment.")
    write_json(directory / "resolved_config.json", cfg)
    write_json(directory / "data_provenance.json", provenance)
    representation = NormalRepresentation(cfg["model"]).to(device)
    physical = cfg["runtime"].get("cuda_visible_devices") if device.type == "cuda" else None
    print(f"{entity.dataset}/{entity.name} seed={seed} physical_gpu={physical} device={device} W={w} C={train.z.shape[1]}", flush=True)
    if stage in {"all", "representation"} and checkpoint is None:
        fit_representation(representation, train, valid, cfg, directory, normal_key)
    if checkpoint is None:
        checkpoint = _load_checkpoint(representation_path, normal_key, device)
    representation.load_state_dict(checkpoint["model"])
    representation.eval()
    for parameter in representation.parameters():
        parameter.requires_grad_(False)
    if stage == "representation":
        return directory
    if representation_key is None:
        representation_key = fingerprint({"normal": normal_key, "checkpoint": file_sha256(representation_path)})
    if selected_bundle is None and (detector_dir / "bundle.pt").exists():
        saved = load_torch_checkpoint(detector_dir / "bundle.pt", map_location=device)
        identity = saved.get("experiment") or {}
        if identity.get("normal_key") == normal_key and identity.get("representation_key") == representation_key:
            selected_bundle = saved
    detector_settings = {k: cfg["stage2"].get(k) for k in ("variant", "fixed_retention", "sources")}
    detector_key = fingerprint({"representation": representation_key, "stage2": cfg["stage2"], "seed": seed})
    experiment = {"normal_key": normal_key, "representation_key": representation_key, "detector_key": detector_key,
        "implementation_key": code_key, "packages": runtime_packages, "data_provenance": provenance}
    if selected_bundle is not None:
        saved_identity = selected_bundle["experiment"]
        if saved_identity.get("detector_key") != detector_key:
            raise ValueError("Completed detector parameters differ; start a new run for changed training settings.")
        if stage in {"all", "evaluate", "calibrate", "detector"}:
            if stage == "detector":
                return detector_dir
            detector = DualTrajectoryDetector(representation.history.identity.weight, **detector_settings).to(device)
            detector.load_state_dict(selected_bundle["detector"])
            detector.eval()
            experiment = saved_identity
            calibration = experiment["calibration"]
            compatible = (experiment.get("calibration_config") == cfg["calibration"]
                          and experiment.get("execution") == cfg["runtime"].get("detector_execution", "block"))
            if stage == "calibrate":
                calibration, _ = _calibrate_model(entity, train, valid, representation, detector, cfg,
                    directory, representation_key, fingerprint({"model": detector_key, "calibration": cfg["calibration"],
                                                               "execution": cfg["runtime"].get("detector_execution", "block")}))
                experiment.update(calibration=calibration, calibration_config=copy.deepcopy(cfg["calibration"]),
                                  execution=cfg["runtime"].get("detector_execution", "block"), result=None)
                save_bundle(detector_dir / "bundle.pt", representation, detector, scaler, calibration, cfg, provenance, experiment)
                remove_run_result(cfg["runtime"]["run_directory"], entity.dataset, entity.name, seed, variant_name)
                return detector_dir
            if not compatible:
                raise ValueError("Calibration configuration differs; run --stage calibrate before evaluation.")
            previous = experiment.get("result")
            same_metrics = previous and previous.get("metrics_protocol") == METRICS_PROTOCOL and (
                set(previous["metrics"].get("score", {})) == set(cfg["evaluation"]["score_metrics"]))
            if stage == "all" and same_metrics:
                csv_path = write_run_result(cfg["runtime"]["run_directory"], previous)
                if not cfg["runtime"].get("keep_intermediates", False):
                    prune_intermediates(directory, detector_dir)
                print(f"completed model reused; metrics {csv_path}", flush=True)
                return detector_dir
            return _evaluate_and_save(entity, train, valid, raw_test, labels, scaler, provenance, cfg, seed,
                representation, detector, directory, detector_dir, calibration, experiment, diagnostics)
    cache_dir = directory / "task_cache"
    if stage in {"all", "cache"}:
        cache_key, _, _ = build_task_cache(representation, train, valid, cfg, cache_dir, seed, representation_key)
    elif stage == "detector":
        manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError("Task cache is incomplete; run --stage cache.")
        expected_settings = {k: cfg["stage2"][k] for k in ("families", "train_pairs", "validation_pairs", "validation_seeds")}
        expected_key = fingerprint({"version": 1, "representation": representation_key, "seed": seed, "tasks": expected_settings})
        if manifest["key"] != expected_key:
            raise ValueError("Task cache does not match the frozen representation and task settings.")
        cache_key = manifest["key"]
    else:
        cache_key = None
    if stage == "cache":
        return cache_dir
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_path = detector_dir / "detector.pt"
    detector_history = detector_dir / "stage2_history.json"
    if detector_path.exists():
        _load_checkpoint(detector_path, detector_key, device)
    write_json(detector_dir / "resolved_config.json", cfg)
    # Initialization must not depend on whether earlier stages were loaded or trained in this process.
    seed_everything(seed + 10000)
    detector = DualTrajectoryDetector(representation.history.identity.weight, **detector_settings).to(device)
    completed = detector_path.exists() and detector_history.exists()
    if completed:
        history_record = json.loads(detector_history.read_text(encoding="utf-8"))
        completed = history_record.get("complete", True) and history_record.get("key", detector_key) == detector_key
    if stage in {"all", "detector"} and not completed:
        if detector_history.exists():
            detector_history.unlink()
        fit_detector(detector, representation, cfg, cache_dir, cache_key, detector_dir, detector_key, seed)
    elif not completed:
        raise ValueError("Detector training is incomplete; run --stage detector before calibration/evaluation.")
    detector.load_state_dict(_load_checkpoint(detector_path, detector_key, device)["model"])
    detector.eval()
    if stage == "detector":
        return detector_dir
    calibration_path = detector_dir / "calibration.json"
    calibration_identity = {"detector": file_sha256(detector_path), "data": normal_key, "calibration": cfg["calibration"]}
    if cfg["runtime"].get("detector_execution", "block") == "block":
        calibration_identity["execution"] = "block"
    cal_key = fingerprint(calibration_identity)
    if stage in {"all", "calibrate"}:
        calibration, normal_scores = _calibrate_model(entity, train, valid, representation, detector, cfg,
                                                     directory, representation_key, cal_key)
        write_json(calibration_path, calibration)
        if cfg["runtime"].get("keep_intermediates", False):
            np.savez_compressed(detector_dir / "normal_validation_scores.npz", scores=normal_scores)
        experiment.update(calibration=calibration, calibration_config=copy.deepcopy(cfg["calibration"]),
                          execution=cfg["runtime"].get("detector_execution", "block"), result=None)
        save_bundle(detector_dir / "bundle.pt", representation, detector, scaler, calibration, cfg, provenance, experiment)
    else:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration["key"] != cal_key:
            raise ValueError("Calibration differs from the detector or configuration; run --stage calibrate.")
    if stage == "calibrate":
        return detector_dir
    return _evaluate_and_save(entity, train, valid, raw_test, labels, scaler, provenance, cfg, seed,
        representation, detector, directory, detector_dir, calibration, experiment, diagnostics)
