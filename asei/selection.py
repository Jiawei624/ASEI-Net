"""Normal-only chronological audit and synthetic ranking for detector selection."""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .batching import feature_blocks, tensor_bytes
from .calibration import ParallelCalibration
from .data import Record
from .features import forecast_errors
from .parallel import preparation_workers, prefetch_depth


class NormalSelectionReplay:
    """A bounded contiguous selection suffix; final calibration is never read."""
    def __init__(self, model, record, cfg):
        self.cfg = cfg
        window = cfg["model"]["window"]
        target = max(record.target_start, window, len(record.z) - cfg["stage2"]["selection_normal_points"])
        start = max(0, target - window)
        self.record = Record(record.z[start:].copy(), record.mask[start:].copy(), target - start)
        self.errors, self.ready, _ = forecast_errors(self.record, model.normal, cfg["runtime"]["feature_batch_size"],
            log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 30), workers=preparation_workers(cfg), prefetch=prefetch_depth(cfg))
        self.record.routing = model.routing(self.record, cfg["stage2"]["block_length"])
        self.prepared_cache, self.cache_bytes = {}, 0
        self.cache_limit = cfg["runtime"].get("normal_selection_cache_mb", 0) * 1024 ** 2

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        record, cfg = self.record, self.cfg
        state, arrays = None, {}
        for offset, _, f in feature_blocks(model.normal.schema, [record], [0], len(record.z),
                cfg["stage2"]["block_length"], cfg, [self.errors], [self.ready]):
            device = model.normal.schema.amp_mean.device
            if offset in self.prepared_cache:
                prepared = {k: v.to(device) for k, v in self.prepared_cache[offset].items()}
            else:
                prepared = model.frozen_block(f, None if state is None else state.normal)
                keep = {k: prepared[k] for k in ("feature", "normal", "reference_input", "actual_projected", "reference_valid",
                        "short_energy", "short_available", "final_normal")}
                size = tensor_bytes(keep)
                if self.cache_bytes + size <= self.cache_limit:
                    self.prepared_cache[offset] = {k: v.detach().cpu() for k, v in keep.items()}
                    self.cache_bytes += size
            values, state = model.forward_block(f, state, prepared=prepared)
            for key in ("short", "dynamic", "short_valid", "dynamic_valid"):
                arrays.setdefault(key, []).append(values[key][0].cpu().numpy())
        raw = {k: np.concatenate(v)[record.target_start:] for k, v in arrays.items()}
        cut = int(len(raw["short"]) * cfg["stage2"]["selection_reference_fraction"])
        info = dict(normal_reference_points=cut, normal_audit_points=len(raw["short"]) - cut,
                    normal_constraint_supported=False, normal_audit_scope="selection_contiguous_suffix")
        if min(cut, info["normal_audit_points"]) < 1:
            return None, info
        references = [raw[k][:cut][raw[k + "_valid"][:cut]] for k in ("short", "dynamic")]
        weights = [0., 1.] if model.variant == "no_energy" else [1., 0.] if model.variant == "energy_only" else cfg["calibration"]["weights"]
        if weights[1] == 0:
            info["normal_constraint_reason"] = "inactive_dynamic_branch"
            return None, info
        if any(weight > 0 and not len(v) for weight, v in zip(weights, references)):
            return None, info
        # Preserve the original selection protocol. Final evaluation may use a
        # continuous ranking map, but it never changes this internal tail rank.
        calibration = ParallelCalibration(*references, weights, cfg["calibration"]["alpha"])
        if not all(v["supported"] for weight, v in zip(weights, calibration.resolution().values()) if weight > 0):
            return None, info
        output = calibration.transform(raw["short"][cut:], raw["dynamic"][cut:],
                                       raw["short_valid"][cut:], raw["dynamic_valid"][cut:])
        usable = raw["dynamic_valid"][cut:]
        if not usable.any():
            return None, info
        dynamic_alarm = output["p_dynamic"] <= calibration.alpha * calibration.weights[1]
        info.update(normal_constraint_supported=True, normal_dynamic_fpr=float(dynamic_alarm[usable].mean()),
                    normal_fused_fpr=float(output["labels"][output["valid"]].mean()))
        return calibration, info


def constrained_selection(validation, raw, calibration, audit, settings):
    """FPR bands take precedence; AP and phase-balanced coverage select within a band."""
    result = dict(audit)
    if calibration is None:
        result.update(objective=validation["loss"], selection_fallback=audit.get("normal_constraint_reason", "insufficient_normal_rank_resolution"))
        return result
    output = calibration.transform(raw["short"], raw["dynamic"], raw["short_valid"], raw["dynamic_valid"])
    eligible = output["valid"] & (raw["weight"] > 0)
    truth = raw["truth"][eligible] > 0
    if not truth.any() or truth.all():
        raise ValueError("Synthetic selection requires observable normal and anomalous locations.")
    weight = raw["weight"][eligible]
    ap = float(average_precision_score(truth, output["scores"][eligible], sample_weight=weight))
    coverage = float(np.sum(weight * truth * output["labels"][eligible]) / np.sum(weight * truth))
    mix = settings["selection_coverage_weight"]
    quality = (1 - mix) * ap + mix * coverage
    excess = max(0., audit["normal_dynamic_fpr"] - settings["selection_fpr_limit"])
    band = max(0, math.ceil(excess / settings["selection_fpr_tolerance"] - 1e-12))
    result.update(objective=float(band + 1 - quality), synthetic_ap=ap, positive_coverage=coverage,
                  selection_quality=float(quality), normal_fpr_band=band)
    return result
