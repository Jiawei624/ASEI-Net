"""End-to-end detector training, held-out calibration and causal deployment."""
from __future__ import annotations

import copy
from pathlib import Path
import time

import numpy as np
import torch

from .compat import load_torch_checkpoint
from .config import (configure_cuda, fingerprint, implementation_fingerprint, resolve_config,
                      resolve_paths, write_json)
from .data import Record, Standardizer, load_entity, validation_context
from .metrics import (METRICS_PROTOCOL, _evaluation_scores, label_metrics, score_metrics,
                       event_metrics, evaluate_anomaly_ratios, thresholds_from_anomaly_ratios)
from .resources import configure_cpu, cooperate
from .results import write_run_result
from .runs import training_root
from . import FORMAT, METHOD
from .calibration import ParallelCalibration
from .cache import prune_task_cache
from .features import DescriptorSchema, forecast_errors
from .model import DualTrajectoryDetector, TrajectoryState
from .normal import NormalModel
from .tasks import ResidualLagTasks
from .training import (_save, fit_forecaster, fit_teacher, fit_energy_scales,
                       build_pairs, seed_everything)
from .detection_training import fit_detector
from .progress import Progress


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


@torch.no_grad()
def score_raw(model, record, cfg, start=0, state=None, diagnostics=False):
    """Past-only forecasting is batched; reference/state updates remain chronological."""
    model.eval()
    interval = cfg["runtime"].get("log_interval_seconds", 30)
    errors, ready, _ = forecast_errors(record, model.normal, cfg["runtime"]["feature_batch_size"],
                                      log_interval_seconds=interval)
    block = 1 if cfg["runtime"].get("detector_execution") == "step" else cfg["stage2"]["block_length"]
    outputs = {}
    progress = Progress("scoring blocks", (len(record.z) - start + block - 1) // block, interval) if len(record.z) - start > block else None
    for begin in range(start, len(record.z), block):
        cooperate()
        features = model.normal.schema.block([record], [begin], min(block, len(record.z) - begin), [errors], [ready])
        values, state = model.forward_block(features, state, return_aux=diagnostics)
        for key, value in values.items():
            if not diagnostics or key not in {"normal", "observed", "reference_pair"}:
                outputs.setdefault(key, []).append(value[0].cpu().numpy())
        if progress:
            progress.update((begin - start) // block + 1)
    if not outputs:
        raise ValueError("No time points to score.")
    return {key: np.concatenate(value) for key, value in outputs.items()}, state.detach()


def _apply(calibration, raw, alpha=None):
    return calibration.transform(raw["short"], raw["dynamic"], raw["short_valid"], raw["dynamic_valid"], alpha)


def _join(*records):
    return Record(np.concatenate([r.z for r in records]), np.concatenate([r.mask for r in records]))


def _bootstrap(record, state, window):
    n = max(window, 32)
    return dict(z=record.z[-n:].copy(), mask=record.mask[-n:].copy(),
                normal=state.normal.cpu(), observed=state.observed.cpu())


def save_bundle(path, model, scaler, calibration, cfg, provenance, experiment, bootstrap=None):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    _save(path, dict(format=FORMAT, schema=model.normal.metadata, config=copy.deepcopy(cfg), model=state,
                     scaler=scaler.as_dict(), calibration=calibration.as_dict(), provenance=provenance,
                     experiment=experiment, bootstrap=bootstrap))


def _load_bundle(saved, device="cpu", alpha=None):
    if saved.get("format") != FORMAT:
        raise ValueError("Unsupported detector bundle format.")
    cfg = saved["config"]
    normal = NormalModel(saved["schema"], cfg).to(device)
    model = DualTrajectoryDetector(normal, cfg["stage2"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    calibration = ParallelCalibration.from_dict(saved["calibration"])
    if alpha is not None:
        if not 0 < alpha < 1:
            raise ValueError("Prediction alpha must be in (0,1).")
        calibration.alpha = float(alpha)
    scaler = Standardizer(np.asarray(saved["scaler"]["mean"]), np.asarray(saved["scaler"]["scale"]))
    return StreamingDetector(model, scaler, calibration, cfg, saved.get("bootstrap")), saved["provenance"]["channels"]


def load_bundle(path, device="cpu", anomaly_ratio=None, alpha=None):
    if anomaly_ratio is not None:
        raise ValueError("Use alpha for deployment; anomaly ratios are offline evaluation only.")
    device = get_device(device)
    return _load_bundle(load_torch_checkpoint(path, map_location=device), device, alpha)


class StreamingDetector:
    def __init__(self, model, scaler, calibration, cfg, bootstrap=None):
        self.model, self.scaler, self.calibration, self.cfg = model, scaler, calibration, cfg
        self.bootstrap = bootstrap
        self.reset(use_bootstrap=True)

    def reset(self, use_bootstrap=False):
        c = self.model.normal.schema.channels
        device = self.model.normal.schema.amp_mean.device
        self.context = Record(np.empty((0, c), np.float32), np.empty((0, c), bool))
        self.state = self.model.initial_state(1)
        self.previous, self.index = None, 0
        if use_bootstrap and self.bootstrap is not None:
            def array(value, dtype):
                if torch.is_tensor(value):
                    value = value.cpu().numpy()
                return np.asarray(value, dtype).copy()
            self.context = Record(array(self.bootstrap["z"], np.float32), array(self.bootstrap["mask"], bool))
            self.state = TrajectoryState(self.bootstrap["normal"].to(device), self.bootstrap["observed"].to(device))
            self.previous = self.context.z[-1].astype(np.float64) * self.scaler.scale + self.scaler.mean

    @torch.no_grad()
    def push_many(self, samples):
        samples = np.asarray(samples, np.float64)
        if samples.ndim != 2 or samples.shape[1] != self.model.normal.schema.channels:
            raise ValueError("Samples must be [time, fitted channels].")
        rows = []
        # Bounded blocks preserve identical semantics for files and live samples.
        for begin in range(0, len(samples), self.cfg["stage2"]["block_length"]):
            incoming = self.scaler.transform(samples[begin:begin + self.cfg["stage2"]["block_length"]], self.previous)
            record, start = _join(self.context, incoming), len(self.context.z)
            raw, self.state = score_raw(self.model, record, self.cfg, start, self.state)
            result = _apply(self.calibration, raw)
            for k in range(len(incoming.z)):
                valid = bool(result["valid"][k])
                rows.append(dict(index=self.index, valid=valid, score=float(result["scores"][k]) if valid else None,
                                 alarm=bool(result["labels"][k]), p_mix=float(result["p_mix"][k]),
                                 p_short=float(result["p_short"][k]), p_dynamic=float(result["p_dynamic"][k]),
                                 short=float(raw["short"][k]) if raw["short_valid"][k] else None,
                                 dynamic=float(raw["dynamic"][k]) if raw["dynamic_valid"][k] else None))
                self.index += 1
            keep = max(self.model.normal.predictor.window, 32)
            self.context = Record(record.z[-keep:].copy(), record.mask[-keep:].copy())
            self.previous = record.z[-1].astype(np.float64) * self.scaler.scale + self.scaler.mean
        return rows

    def push(self, sample):
        return self.push_many(np.asarray(sample)[None])[0]


def _weights(cfg):
    variant = cfg["stage2"]["variant"]
    # These are declared ablations, never data-dependent reallocation of a missing branch.
    return [0., 1.] if variant == "no_energy" else [1., 0.] if variant == "energy_only" else cfg["calibration"]["weights"]


def _evaluate(raw, labels, calibration, cfg):
    output = _apply(calibration, raw)
    actual, _, valid, full_scores = _evaluation_scores(labels, output["scores"], output["valid"])
    warnings = {}
    if valid.any():
        ranking = score_metrics(actual, full_scores, cfg["evaluation"]["score_metrics"], warnings)
    else:
        ranking = {name: float("nan") for name in cfg["evaluation"]["score_metrics"]}
        warnings["scores"] = "No scorable test point."
    metrics = dict(protocol=METRICS_PROTOCOL, score=ranking, label_by_alpha=[],
        comparison="p_mix <= alpha", point_adjusted_metrics="adjust_* fields are compatibility outputs only")
    alarms = []
    for alpha in cfg["calibration"]["alphas"]:
        predicted = valid & (output["p_mix"] <= alpha)
        events = event_metrics(actual, predicted, valid, cfg["evaluation"]["recovery_confirmation"])
        entries = label_metrics(actual, predicted, cfg["evaluation"]["label_metrics"], warnings, full_scores)
        if not valid.any() and "precision_at_k" in entries:
            entries["precision_at_k"] = float("nan")
        negatives = actual == 0
        # Include unavailable negatives in full-record FPR, and disclose coverage separately.
        fpr = float(predicted[negatives].mean()) if negatives.any() else None
        covered = negatives & valid
        scored_fpr = float(predicted[covered].mean()) if covered.any() else None
        metrics["label_by_alpha"].append(dict(alpha=alpha, threshold=float(-np.log(alpha)), metrics=entries, event=events,
            normal_fpr=fpr, scorable_normal_fpr=scored_fpr, scorable_fraction=float(valid.mean()),
            resolution=calibration.resolution(alpha)))
        alarms.append(predicted)
    metrics["undefined_or_warnings"] = warnings
    return metrics, output, np.stack(alarms)


def run_entity(entity, config, seed, stage="all", diagnostics=False):
    cfg = resolve_paths(copy.deepcopy(config))
    if diagnostics:
        cfg["runtime"]["keep_intermediates"] = True
    configure_cuda(cfg)
    configure_cpu(cfg)
    seed_everything(seed)
    device = get_device(cfg["runtime"]["device"])
    fit, heldout, test, labels, scaler, provenance = load_entity(entity, cfg["data"]["train_fraction"])
    cfg = resolve_config(cfg, len(fit.z), fit.z.shape[1])
    cut = int(len(heldout.z) * cfg["data"]["selection_fraction"])
    if min(cut, len(heldout.z) - cut) < 4:
        raise ValueError("Normal selection and calibration segments each need at least four samples.")
    independent = bool(entity.validation_path)
    select_raw = Record(heldout.z[:cut], heldout.mask[:cut])
    selection = validation_context(fit, select_raw, cfg["model"]["window"], independent)
    if len(selection.z) - max(selection.target_start, cfg["model"]["window"]) < 10:
        raise ValueError("Normal selection has too few targets after causal warm-up; provide more normal validation data.")
    directory = training_root(cfg, stage) / entity.dataset / entity.name / f"seed_{seed}"
    detector_dir = directory / "detectors" / cfg["stage2"]["variant"]
    detector_dir.mkdir(parents=True, exist_ok=True)
    print(f"entity={entity.dataset}/{entity.name} seed={seed} device={device} "
          f"fit={len(fit.z)} selection={cut} calibration={len(heldout.z) - cut} channels={fit.z.shape[1]}", flush=True)
    implementation = implementation_fingerprint()
    normal_key = fingerprint(dict(data=provenance, model=cfg["model"], stage1=cfg["stage1"],
        memory=cfg["memory"], types=cfg["data"].get("channel_types"), implementation=implementation, seed=seed,
        discrete_max_values=cfg["stage2"]["discrete_max_values"], selection_points=cut,
        relation_l1=cfg["stage2"]["relation_l1"], initial_teacher_cell=[cfg["stage2"]["eta"], cfg["stage2"]["rho"]]))
    teacher_key = fingerprint(dict(normal=normal_key, teacher=cfg["teacher"],
        effective_block=min(cfg["stage2"]["block_length"], cfg["teacher"]["block_length"]),
        settings={k: cfg["stage2"][k] for k in ("eta", "rho", "reference_trim", "reference_prior")}))
    detector_key = fingerprint(dict(teacher=teacher_key, stage2=cfg["stage2"], seed=seed))
    cfg.update(_normal_key=normal_key, _teacher_key=teacher_key)
    write_json(detector_dir / "resolved_config.json", cfg)
    experiment = dict(normal_key=normal_key, teacher_key=teacher_key, detector_key=detector_key,
        implementation_key=implementation, timings={}, histories={})
    bundle = detector_dir / "bundle.pt"
    if stage in {"evaluate", "calibrate"} and bundle.exists():
        saved = load_torch_checkpoint(bundle, map_location=device)
        if saved["experiment"]["detector_key"] != detector_key:
            raise ValueError("Bundle training identity differs; use its saved configuration or a new run.")
        stream, _ = _load_bundle(saved, device)
        model, experiment = stream.model, saved["experiment"]
    else:
        detector_path = detector_dir / "detector.pt"
        saved_detector = load_torch_checkpoint(detector_path, map_location=device) if detector_path.exists() else None
        detector_complete = saved_detector is not None and saved_detector.get("complete", False)
        if stage in {"evaluate", "calibrate"} and not detector_complete:
            raise FileNotFoundError("Train the detector before calibration/evaluation.")
        schema = DescriptorSchema.fit(fit, cfg["stage2"]["discrete_max_values"], cfg["data"].get("channel_types"), provenance["channels"])
        normal = NormalModel(schema.metadata, cfg).to(device)
        model = DualTrajectoryDetector(normal, cfg["stage2"]).to(device)
        start = time.perf_counter()
        experiment["histories"]["forecast"] = fit_forecaster(normal, fit, selection, cfg, directory, normal_key, seed)
        experiment["timings"]["forecast_seconds"] = time.perf_counter() - start
        if stage == "representation":
            return directory
        seed_everything(seed + 7000)
        start = time.perf_counter()
        experiment["histories"]["teacher"] = fit_teacher(model, fit, selection, cfg, directory, teacher_key, seed)
        experiment["timings"]["teacher_seconds"] = time.perf_counter() - start
        if stage == "teacher":
            return directory
        scale_path = detector_dir / "energy_scales.pt"
        if scale_path.exists():
            scales = load_torch_checkpoint(scale_path, map_location=device)
            if scales["key"] != detector_key:
                raise ValueError("Energy-scale identity differs; use a new run.")
            normal.memory.energy_scale.copy_(scales["value"])
            experiment["energy_scales"] = scales["statistics"]
        else:
            start = time.perf_counter()
            experiment["energy_scales"] = fit_energy_scales(model, fit, cfg)
            experiment["timings"]["energy_scale_seconds"] = time.perf_counter() - start
            _save(scale_path, dict(key=detector_key, value=normal.memory.energy_scale.cpu(), statistics=experiment["energy_scales"]))
        seed_everything(seed + 10000)
        model = DualTrajectoryDetector(normal, cfg["stage2"]).to(device)
        if not detector_complete or stage == "cache":
            relation = (ResidualLagTasks(fit, normal.metadata["supports"], normal.predictor.weights().detach().cpu().numpy(), seed)
                        if cfg["stage2"]["relation_generator"] == "residual_var" else None)
            start = time.perf_counter()
            train_cache = valid_cache = None
            try:
                train_cache = build_pairs(model, fit, fit, cfg, directory, "train", seed + 20000, provenance["channels"], relation)
                valid_cache = build_pairs(model, selection, selection, cfg, directory, "selection", seed + 30000, provenance["channels"], relation)
                experiment["timings"]["pair_preparation_seconds"] = time.perf_counter() - start
                if stage == "cache":
                    return directory
                start = time.perf_counter()
                experiment["histories"]["detector"] = fit_detector(model, train_cache, valid_cache, cfg, detector_dir, detector_key, seed)
                experiment["timings"]["detector_seconds"] = time.perf_counter() - start
            finally:
                if train_cache is not None:
                    train_cache.close()
                if valid_cache is not None:
                    valid_cache.close()
        else:
            saved = saved_detector
            if saved["key"] != detector_key:
                raise ValueError("Detector checkpoint identity differs; use a new run.")
            model.load_state_dict(saved["model"])
            experiment["histories"]["detector"] = saved["history"]
        if stage == "detector":
            return detector_dir
    model.eval().requires_grad_(False)
    # The full frozen replay uses the actual past; only the held-out suffix enters the reference.
    official = fit if independent else _join(fit, heldout)
    calibration_record = heldout if independent else official
    calibration_start = cut if independent else len(fit.z) + cut
    start = time.perf_counter()
    normal_raw, final_state = score_raw(model, calibration_record, cfg)
    selected = {k: v[calibration_start:] for k, v in normal_raw.items()}
    calibration = ParallelCalibration(selected["short"][selected["short_valid"]],
        selected["dynamic"][selected["dynamic_valid"]], _weights(cfg), cfg["calibration"]["alpha"])
    experiment["timings"]["normal_replay_seconds"] = time.perf_counter() - start
    experiment["calibration_scope"] = dict(source="independent_normal" if independent else "official_normal_suffix",
        fit_points=len(fit.z), selection_points=cut, calibration_points=len(heldout.z) - cut,
        calibration_start=calibration_start, weights=_weights(cfg), resolution=calibration.resolution())
    bootstrap = None
    if entity.test_continuous:
        if independent:
            _, final_state = score_raw(model, official, cfg)
        bootstrap = _bootstrap(official, final_state, cfg["model"]["window"])
    save_bundle(bundle, model, scaler, calibration, cfg, provenance, experiment, bootstrap)
    write_json(detector_dir / "calibration.json", {**experiment["calibration_scope"], "method": "parallel_normal",
               "alpha": calibration.alpha, "alphas": cfg["calibration"]["alphas"]})
    if stage == "calibrate":
        return detector_dir
    start = time.perf_counter()
    if bootstrap is None:
        raw, _ = score_raw(model, test, cfg, diagnostics=diagnostics)
    else:
        context = Record(bootstrap["z"], bootstrap["mask"])
        state = TrajectoryState(bootstrap["normal"].to(device), bootstrap["observed"].to(device))
        raw, _ = score_raw(model, _join(context, test), cfg, len(context.z), state, diagnostics)
    experiment["timings"]["test_scoring_seconds"] = time.perf_counter() - start
    print(f"evaluation label_metrics={len(cfg['evaluation']['label_metrics'])} "
          f"score_metrics={len(cfg['evaluation']['score_metrics'])}", flush=True)
    metrics, output, alarms = _evaluate(raw, labels, calibration, cfg)
    result = dict(dataset=entity.dataset, entity=entity.name, seed=seed, method=METHOD, variant=model.variant,
        run_id=cfg["runtime"]["run_id"], run_created_ns=cfg["runtime"]["run_created_ns"],
        model_key=detector_key, implementation_key=implementation, metrics_protocol=METRICS_PROTOCOL,
        model_file=bundle.relative_to(Path(cfg["runtime"]["run_directory"])).as_posix(), test_points=len(labels),
        calibration={**experiment["calibration_scope"], "method": "parallel_normal", "alpha": calibration.alpha},
        metrics=metrics, config=cfg, provenance=provenance, timing=experiment["timings"],
        model_analysis=dict(contraction_bound=model.normal.teacher.cell.contraction,
            hidden_spectral_norm=float(torch.linalg.svdvals(model.normal.teacher.cell.hidden.weight)[0].cpu()),
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_detector_parameters=sum(p.numel() for name, p in model.named_parameters() if not name.startswith("normal."))),
        environment=dict(torch=str(torch.__version__), numpy=np.__version__, device=str(device)))
    # Keep TAB thresholds visibly separate from the operational normal-only protocol.
    ratios = cfg["calibration"].get("offline_anomaly_ratios")
    if ratios:
        fit_scores = _apply(calibration, {k: v[:len(fit.z)] for k, v in normal_raw.items()})["scores"]
        if independent:
            fit_raw, _ = score_raw(model, fit, cfg)
            fit_scores = _apply(calibration, fit_raw)["scores"]
        tab_cal = thresholds_from_anomaly_ratios(fit_scores, output["scores"][output["valid"]], ratios, ratios[0])
        tab, _ = evaluate_anomaly_ratios(labels, output["scores"], output["valid"], tab_cal, cfg["evaluation"])
        write_json(detector_dir / "tab_offline.json", dict(protocol="offline_test_distribution", metrics=tab))
    experiment["result"] = result
    save_bundle(bundle, model, scaler, calibration, cfg, provenance, experiment, bootstrap)
    write_json(detector_dir / "result.json", result)
    csv_path = write_run_result(cfg["runtime"]["run_directory"], result)
    if cfg["runtime"]["keep_intermediates"] or diagnostics:
        np.savez_compressed(detector_dir / "predictions.npz", labels=labels, alarms_by_alpha=alarms,
            alphas=np.asarray(cfg["calibration"]["alphas"]),
            **{k: v for k, v in output.items() if k != "labels"},
            **{"raw_" + k: v for k, v in raw.items()})
    else:
        prune_task_cache(directory)
    print(f"saved metrics {csv_path}; best model {bundle}", flush=True)
    return detector_dir
