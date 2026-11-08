"""Defaults and validation for the normal-reference detector."""
from __future__ import annotations

import math

from . import METHOD


VARIANTS = {"full", "mlp", "single_gru", "no_alignment", "zero_alignment", "fixed_gate",
            "no_energy", "energy_only", "no_tail", "no_state", "no_external"}


def defaults(cfg):
    sections = {
        "data": dict(root="dataset", train_fraction=.6, selection_fraction=.5, channel_types=None,
                     datasets=["PSM", "SMD", "SWAT", "PUMP"], multivariate_only=True,
                     use_prepared_manifest=True, manifest=None, archive="TAB_dataset.zip"),
        "model": dict(window="auto", d_model="auto", heads=4, layers=2, dropout=.1,
                      marginal_prototypes="auto", joint_prototypes="auto"),
        "stage1": dict(epochs=80, batch_size=64, lr=.0003, weight_decay=.0001, patience=10,
                       stride=1, max_train_points=0, max_validation_points=4096),
        "teacher": dict(epochs=10, patience=3, lr=.0003, batch_sequences=8, sequence_length=256,
                        train_sequences=256, block_length=64, optimizer_update="task_batch"),
        "stage2": dict(method=METHOD, variant="full", epochs=50, patience=8, batch_pairs=8,
                       lr=.0003, block_length=128, train_pairs="auto", validation_pairs=128,
                       families=["level", "variance", "temporal", "relation"],
                       fixed_retention=.9, eta=.2, rho=.95, reference_trim=.1, reference_prior=.25,
                       lambda_state=.1, lambda_tail=.1, tail_fraction=.01, tail_margin=.5,
                       optimizer_update="task_batch", hard_negative_fraction=.25, discrete_max_values=16, donor_candidates=16,
                       boundary_blend=4, relation_l1=.001, relation_generator="residual_var"),
        "memory": dict(pool_size=8192, prototype_chunk=32, reference_block_length=16, rare_fraction=.25),
        "calibration": dict(method="parallel_normal", alpha=.01, alphas=[.001, .005, .01], weights=[.5, .5]),
        "evaluation": dict(skip_vus=False, recovery_confirmation=8, regression_metrics=[],
                           supplementary_label_metrics=[], latency_points=0),
        "runtime": dict(device="auto", output="outputs", log_dir="logs", feature_batch_size=64,
                        threads=4, interop_threads=1, cpu_cores=8, cpu_max_percent=80,
                        seeds=[41, 42, 43, 44, 45], keep_intermediates=False,
                        preparation_workers=2, prefetch_batches=2,
                        task_cache_memory_mb=256, log_interval_seconds=30, detector_execution="block"),
    }
    for section, values in sections.items():
        target = cfg.setdefault(section, {})
        for key, value in values.items():
            target.setdefault(key, value)
    from .metrics import LABEL_METRIC_ORDER, SCORE_METRIC_ORDER
    cfg["evaluation"]["label_metrics"] = list(LABEL_METRIC_ORDER)
    cfg["evaluation"]["score_metrics"] = [name for name in SCORE_METRIC_ORDER
        if not (cfg["evaluation"]["skip_vus"] and name.startswith("VUS_"))]
    if cfg["stage2"]["method"] != METHOD:
        raise ValueError("Unsupported model configuration; use a dataset configuration under configs/.")
    return cfg


def resolve(cfg, n_train, channels):
    defaults(cfg)
    if n_train < 65 or channels < 1:
        raise ValueError("The detector needs at least 65 fitting samples and one observed channel.")
    m, s, cal = cfg["model"], cfg["stage2"], cfg["calibration"]
    for key, value in dict(evidence_transform="identity", evidence_hidden=64, readout_hidden=32,
            channel_embedding_dim=0, lambda_channel=0., onset_fraction=.5, reference_guard_multiplier=0.).items():
        s.setdefault(key, value)
    if s["evidence_transform"] not in {"identity", "signed_log1p"}:
        raise ValueError("stage2.evidence_transform must be identity or signed_log1p.")
    for key in ("evidence_hidden", "readout_hidden", "channel_embedding_dim"):
        value = s[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "channel_embedding_dim" else 1):
            raise ValueError("Invalid stage2." + key)
    if not 0 < s["onset_fraction"] < 1:
        raise ValueError("stage2.onset_fraction must be in (0,1).")
    small = n_train < 8448
    if m["window"] == "auto":
        m["window"] = min(128 if small else 256, ((n_train - 33) // 32) * 32)
    if m["d_model"] == "auto":
        m["d_model"] = 64 if small else 96
    if m["window"] < 32 or m["window"] % 32 or m["window"] >= n_train - 1:
        raise ValueError("window must be a multiple of 32 below fitting length minus one.")
    if m["heads"] < 1 or m["d_model"] < 1 or m["d_model"] % m["heads"]:
        raise ValueError("d_model must be divisible by positive heads.")
    for name, capacity in (("marginal_prototypes", 64 if small else 128), ("joint_prototypes", 128 if small else 256)):
        m[name] = min(capacity if m[name] == "auto" else int(m[name]), n_train - 31)
        if m[name] < 1:
            raise ValueError("Prototype capacities must be positive.")
    m["channels"] = channels
    if s["train_pairs"] == "auto":
        s["train_pairs"] = 1024 if n_train >= 100000 else 512
    if s["variant"] not in VARIANTS:
        raise ValueError("Unknown detector variant.")
    if not s["families"] or set(s["families"]) - {"level", "variance", "temporal", "relation"}:
        raise ValueError("Unknown paired task family.")
    if s["relation_generator"] not in {"residual_var", "matched_donor"}:
        raise ValueError("Unknown relation task generator.")
    if "sources" in s:
        raise ValueError("stage2.sources is unsupported; use a declared detector variant.")
    if s["optimizer_update"] != "task_batch" or cfg["teacher"]["optimizer_update"] != "task_batch":
        raise ValueError("The detector normalizes whole tasks and updates once per task batch.")
    for section, names in (("stage1", ["epochs", "batch_size", "patience", "stride"]),
                           ("teacher", ["epochs", "patience", "batch_sequences", "sequence_length", "train_sequences", "block_length"]),
                           ("stage2", ["epochs", "patience", "batch_pairs", "block_length", "train_pairs", "validation_pairs", "discrete_max_values", "donor_candidates"]),
                           ("memory", ["pool_size", "prototype_chunk", "reference_block_length"]),
                           ("runtime", ["feature_batch_size", "threads"]),
                           ("model", ["heads", "layers", "d_model", "window"])):
        for key in names:
            value = cfg[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(section + "." + key + " must be a positive integer.")
    for section, key in (("stage2", "boundary_blend"), ("stage1", "max_train_points"), ("stage1", "max_validation_points")):
        if isinstance(cfg[section][key], bool) or not isinstance(cfg[section][key], int) or cfg[section][key] < 0:
            raise ValueError(section + "." + key + " must be a nonnegative integer.")
    for section in ("stage1", "teacher", "stage2"):
        if not math.isfinite(cfg[section]["lr"]) or cfg[section]["lr"] <= 0:
            raise ValueError(section + ".lr must be finite and positive.")
        options = cfg[section]
        decay = options.get("weight_decay", 0.)
        patience = options.get("lr_patience", 0)
        if not math.isfinite(decay) or decay < 0 or isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError("Invalid weight decay or learning-rate patience in " + section)
        if patience and (not 0 < options.get("lr_factor", .5) < 1 or not 0 < options.get("min_lr", 1e-5) <= options["lr"]):
            raise ValueError("Invalid learning-rate scheduler settings in " + section)
    runtime = cfg["runtime"]
    for key in ("preparation_workers", "prefetch_batches"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("runtime." + key + " must be a nonnegative integer.")
    for key, default in (("preparation_batch_pairs", 8), ("batch_working_memory_mb", 2048)):
        value = runtime.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < (1 if key == "preparation_batch_pairs" else 0):
            raise ValueError("Invalid runtime." + key)
    if not isinstance(runtime.get("save_predictions", False), bool):
        raise ValueError("runtime.save_predictions must be boolean.")
    for key in ("lambda_state", "lambda_tail", "tail_margin", "reference_prior", "relation_l1", "lambda_channel", "reference_guard_multiplier"):
        if not math.isfinite(s[key]) or s[key] < 0:
            raise ValueError("stage2." + key + " must be finite and nonnegative.")
    if not 0 < s["eta"] <= 1 or not 0 < s["rho"] < 1 or not 0 <= s["reference_trim"] < 1:
        raise ValueError("Invalid contraction or reference trimming parameters.")
    if not 0 < s["tail_fraction"] <= 1 or not 0 <= s["hard_negative_fraction"] <= 1 or not 0 <= s["fixed_retention"] <= 1:
        raise ValueError("Invalid tail, task sampling or fixed retention fraction.")
    if not 0 < cfg["data"]["train_fraction"] < 1 or not 0 < cfg["data"]["selection_fraction"] < 1:
        raise ValueError("Fitting and selection fractions must be in (0,1).")
    if not 0 <= cfg["memory"]["rare_fraction"] <= 1:
        raise ValueError("memory.rare_fraction must be in [0,1].")
    if not isinstance(cfg["evaluation"]["recovery_confirmation"], int) or cfg["evaluation"]["recovery_confirmation"] < 1:
        raise ValueError("evaluation.recovery_confirmation must be a positive integer.")
    if not isinstance(cfg["evaluation"]["skip_vus"], bool):
        raise ValueError("evaluation.skip_vus must be boolean.")
    if not 0 <= m["dropout"] < 1 or not math.isfinite(cfg["stage1"]["weight_decay"]) or cfg["stage1"]["weight_decay"] < 0:
        raise ValueError("Invalid dropout or weight decay.")
    if "offline_anomaly_ratios" in cal:
        from .metrics import anomaly_ratios
        cal["offline_anomaly_ratios"] = anomaly_ratios(cal["offline_anomaly_ratios"])
    if cal["method"] != "parallel_normal":
        raise ValueError("The detector uses parallel_normal; put optional TAB ratios in calibration.offline_anomaly_ratios.")
    if not 0 < cal["alpha"] < 1 or not cal["alphas"] or any(not 0 < a < 1 for a in cal["alphas"]):
        raise ValueError("Calibration alpha values must be in (0,1).")
    cal["alphas"] = sorted(set(float(a) for a in cal["alphas"]) | {float(cal["alpha"])})
    weights = cal["weights"]
    if len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights) or abs(sum(weights) - 1) > 1e-8:
        raise ValueError("Predeclared calibration weights must be positive and sum to one.")
    from .metrics import LABEL_METRIC_ORDER, SCORE_METRIC_ORDER
    cfg["evaluation"]["label_metrics"] = list(LABEL_METRIC_ORDER)
    cfg["evaluation"]["score_metrics"] = [n for n in SCORE_METRIC_ORDER if not (cfg["evaluation"]["skip_vus"] and n.startswith("VUS_"))]
    return cfg
