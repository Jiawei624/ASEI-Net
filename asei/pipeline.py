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
from .metrics import evaluate, threshold_from_normal
from .model import NormalRepresentation, DualTrajectoryDetector
from .training import build_task_cache, fit_representation, fit_detector, seed_everything


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
    sequence = extract_features(record, model, cfg["runtime"]["feature_batch_size"])
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


def save_bundle(path, representation, detector, scaler, calibration, cfg, provenance):
    torch.save({"format": 1, "model_config": cfg["model"], "representation": representation.state_dict(),
                "detector": detector.state_dict(),
                "detector_config": {k: cfg["stage2"].get(k) for k in ("variant", "fixed_retention", "sources")},
                "scaler": scaler.as_dict(), "threshold": calibration["threshold"],
                "channel_names": provenance["channels"]}, path)


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


def run_entity(entity: Entity, config: dict, seed: int, stage="all", diagnostics=False):
    config = resolve_paths(copy.deepcopy(config))
    configure_cuda(config)
    seed_everything(seed)
    torch.set_num_threads(config["runtime"]["threads"])
    device = get_device(config["runtime"]["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train, raw_valid, raw_test, labels, scaler, provenance = load_entity(entity, config["data"]["train_fraction"])
    if config["data"].get("multivariate_only", True) and train.z.shape[1] < 2:
        raise ValueError("Multivariate experiments require at least two observed training channels.")
    cfg = resolve_config(config, len(train.z), train.z.shape[1])
    w = cfg["model"]["window"]
    valid = validation_context(train, raw_valid, w, independent=entity.validation_path is not None)
    directory = Path(cfg["runtime"]["output"]) / entity.dataset / entity.name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    code_key = implementation_fingerprint()
    runtime_packages = {name: metadata.version(name) for name in
                        ("numpy", "pandas", "torch", "scikit-learn", "scipy", "statsmodels", "PyYAML")}
    runtime_identity = {"packages": runtime_packages, "python": list(sys.version_info[:2]),
                        "torch_build": str(torch.__version__)}
    normal_key = fingerprint({"implementation": code_key, "model": cfg["model"], "stage1": cfg["stage1"],
                              "data": {k: v for k, v in provenance.items() if k != "source"},
                              "runtime": runtime_identity, "seed": seed})
    write_json(directory / "resolved_config.json", cfg)
    write_json(directory / "data_provenance.json", provenance)
    representation = NormalRepresentation(cfg["model"]).to(device)
    representation_path = directory / "representation.pt"
    print(f"{entity.dataset}/{entity.name} seed={seed} device={device} W={w} C={train.z.shape[1]}", flush=True)
    if stage in {"all", "representation"} and not representation_path.exists():
        fit_representation(representation, train, valid, cfg, directory, normal_key)
    checkpoint = _load_checkpoint(representation_path, normal_key, device)
    representation.load_state_dict(checkpoint["model"])
    representation.eval()
    for parameter in representation.parameters():
        parameter.requires_grad_(False)
    if stage == "representation":
        return directory
    representation_key = fingerprint({"normal": normal_key, "checkpoint": file_sha256(representation_path)})
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
    detector_settings = {k: cfg["stage2"].get(k) for k in ("variant", "fixed_retention", "sources")}
    variant_name = cfg["stage2"]["variant"]
    if cfg["stage2"].get("sources") != ["relation", "marginal", "joint"]:
        variant_name += "_" + ("_".join(cfg["stage2"].get("sources", ["relation", "marginal", "joint"])) or "self")
    detector_dir = directory / "detectors" / variant_name
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_key = fingerprint({"representation": representation_key, "stage2": cfg["stage2"], "seed": seed})
    write_json(detector_dir / "resolved_config.json", cfg)
    # Initialization must not depend on whether earlier stages were loaded or trained in this process.
    seed_everything(seed + 10000)
    detector = DualTrajectoryDetector(representation.history.identity.weight, **detector_settings).to(device)
    detector_path = detector_dir / "detector.pt"
    if stage in {"all", "detector"} and not detector_path.exists():
        fit_detector(detector, representation, cfg, cache_dir, cache_key, detector_dir, detector_key, seed)
    detector.load_state_dict(_load_checkpoint(detector_path, detector_key, device)["model"])
    detector.eval()
    if stage == "detector":
        return detector_dir
    calibration_path = detector_dir / "calibration.json"
    cal_key = fingerprint({"detector": file_sha256(detector_path), "data": normal_key, "calibration": cfg["calibration"]})
    normal_record, validation_start = _normal_calibration_record(train, valid, w, entity.validation_path is not None)
    if stage in {"all", "calibrate"}:
        normal_features, _ = _cached_features(normal_record, representation, cfg,
                                              directory / "normal_features.npz", representation_key + "normal")
        output = score_features(normal_features, representation, detector, cfg["stage2"]["block_length"])
        normal_scores = output["scores"][validation_start:]
        primary = threshold_from_normal(normal_scores, cfg["calibration"]["alpha"])
        calibration = {**primary, "key": cal_key, "validation_targets": len(normal_scores),
                       "scorable_fraction": float(np.isfinite(normal_scores).mean()),
                       "sensitivity": [threshold_from_normal(normal_scores, a) for a in cfg["calibration"]["sensitivity_alphas"]]}
        write_json(calibration_path, calibration)
        np.savez_compressed(detector_dir / "normal_validation_scores.npz", scores=normal_scores)
        save_bundle(detector_dir / "bundle.pt", representation, detector, scaler, calibration, cfg, provenance)
    else:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration["key"] != cal_key:
            raise ValueError("Calibration differs from the detector or configuration; run --stage calibrate.")
    if stage == "calibrate":
        return detector_dir
    if entity.test_continuous:
        # Use only past observed samples. Labels are never passed to this computation.
        past = train if entity.validation_path is not None else normal_record
        test = Record(np.concatenate([past.z, raw_test.z]), np.concatenate([past.mask, raw_test.mask]))
        test_start = len(past.z)
    else:
        test, test_start = raw_test, 0
    features, feature_seconds = _cached_features(test, representation, cfg, directory / "test_features.npz",
                                                 representation_key + "test")
    inference_start = time.perf_counter()
    output = score_features(features, representation, detector, cfg["stage2"]["block_length"], diagnostics)
    scores, ready = output["scores"][test_start:], output["valid"][test_start:]
    inference_seconds = time.perf_counter() - inference_start
    results, alarms = evaluate(labels, scores, ready, calibration["threshold"], cfg["evaluation"])
    sensitivity = []
    for point in calibration["sensitivity"]:
        from .metrics import event_metrics
        sensitivity.append({**point, **event_metrics(labels, ready & (scores > point["threshold"]), ready,
                                                       cfg["evaluation"]["recovery_confirmation"])})
    histories = {}
    for name, file in (("representation", directory / "stage1_history.json"),
                       ("detector", detector_dir / "stage2_history.json"), ("cache", cache_dir / "manifest.json")):
        if file.exists():
            histories[name] = json.loads(file.read_text(encoding="utf-8"))
    timings = {"stage1_seconds": histories.get("representation", {}).get("seconds"),
               "stage2_seconds": histories.get("detector", {}).get("seconds"),
               "cache_preparation_seconds": histories.get("cache", {}).get("preparation_seconds_total"),
               "cache_bytes": histories.get("cache", {}).get("bytes"),
               "test_feature_seconds_this_call": feature_seconds,
               "detector_inference_seconds": inference_seconds,
               "latency": _latency(representation, detector, scaler, calibration["threshold"], raw_test,
                                     cfg["evaluation"]["latency_points"]),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    measured_training = [timings[k] for k in ("stage1_seconds", "stage2_seconds", "cache_preparation_seconds")]
    timings["total_training_seconds"] = sum(measured_training) if all(v is not None for v in measured_training) else None
    prediction_data = {"scores": scores, "valid": ready, "alarms": alarms, "labels": labels,
                       "attribution": output["attribution"][test_start:]}
    if diagnostics:
        prediction_data.update({k: output[k][test_start:] for k in
                                ("retention", "state_difference", "marginal_difference", "joint_difference")})
    np.savez_compressed(detector_dir / "predictions.npz", **prediction_data)
    write_json(detector_dir / "result.json", {"dataset": entity.dataset, "entity": entity.name, "seed": seed,
               "variant": variant_name, "model_key": detector_key, "implementation_key": code_key,
               "test_points": len(labels), "calibration": calibration, "metrics": results,
               "sensitivity": sensitivity, "timing": timings,
               "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                               "platform": platform.platform(), "device": str(device),
                               "cuda": torch.version.cuda,
                               "cudnn": torch.backends.cudnn.version(),
                               "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                               "packages": runtime_packages},
               "parameters": {"representation": sum(p.numel() for p in representation.parameters()),
                              "detector": sum(p.numel() for p in detector.parameters())}, "config": cfg})
    print(f"saved {detector_dir / 'result.json'}", flush=True)
    return detector_dir
