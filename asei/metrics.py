from __future__ import annotations

import warnings

import numpy as np
from sklearn import metrics as skmetrics

from .tasks import event_ranges

LABEL_METRICS = {"accuracy", "precision", "recall", "f_score", "adjust_accuracy", "adjust_precision",
                 "adjust_recall", "adjust_f_score", "rprecision", "rrecall", "rf", "precision_at_k",
                 "affiliation_precision", "affiliation_recall", "affiliation_f", "add"}
SCORE_METRICS = {"auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"}


def threshold_from_normal(scores, alpha=0.01):
    values = np.asarray(scores, np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or not 0 < alpha < 1:
        raise ValueError("Threshold requires finite normal validation scores and alpha in (0,1).")
    # One-based order statistic N - floor(alpha*N), with strictly-greater alarms.
    index = len(values) - int(np.floor(alpha * len(values))) - 1
    threshold = float(np.partition(values, index)[index])
    return {"threshold": threshold, "alpha": alpha, "normal_scores": len(values),
            "normal_exceedance": float(np.mean(values > threshold)), "comparison": ">"}


def add(actual, predicted):
    actual, predicted = _binary_pair(actual, predicted)
    delays = []
    for start, stop in event_ranges(actual):
        hits = np.flatnonzero(predicted[start:stop])
        delays.append(int(hits[0]) if len(hits) else stop - start)
    return float(np.mean(delays)) if delays else float("nan")


def _binary_pair(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Metrics require nonempty, equally sized one-dimensional arrays.")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels and alarms must be binary 0/1.")
    return actual.astype(np.int8), predicted.astype(np.int8)


def event_metrics(actual, predicted, valid=None, confirmation=8):
    actual, predicted = _binary_pair(actual, predicted)
    valid = np.ones(len(actual), bool) if valid is None else np.asarray(valid, bool)
    events = event_ranges(actual)
    coverage, detected, later_coverage, recovery, censored = [], [], [], [], 0
    for i, (start, stop) in enumerate(events):
        coverage.append(float(predicted[start:stop].mean()))
        detected.append(bool(predicted[start:stop].any()))
        later_coverage.append(float(predicted[start + (stop - start) // 2:stop].mean()))
        limit = events[i + 1][0] if i + 1 < len(events) else len(actual)
        found = False
        if limit - stop >= confirmation:
            good = (predicted[stop:limit] == 0) & valid[stop:limit]
            windows = np.convolve(good.astype(np.int64), np.ones(confirmation, np.int64), mode="valid")
            positions = np.flatnonzero(windows == confirmation)
            if len(positions):
                # Report the beginning of the confirmation window, not its end.
                recovery.append(int(positions[0]))
                found = True
        if not found:
            censored += 1
    normal = actual == 0
    ready_normal = normal & valid
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {"events": len(events), "event_detection_rate": mean(detected), "event_coverage": mean(coverage),
            "event_latter_half_coverage": mean(later_coverage), "add": add(actual, predicted),
            "recovery_delay_observed_mean": mean(recovery), "recovery_observed_events": len(recovery),
            "recovery_censored_fraction": censored / len(events) if events else float("nan"),
            "recovery_confirmation": confirmation,
            "normal_false_positive_rate": float(predicted[normal].mean()) if normal.any() else float("nan"),
            "ready_normal_false_positive_rate": float(predicted[ready_normal].mean()) if ready_normal.any() else float("nan"),
            "scorable_fraction": float(valid.mean()),
            "anomaly_scorable_fraction": float(valid[actual == 1].mean()) if actual.any() else float("nan")}


def _safe_metric(name, function, errors):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all="ignore"):
                value = float(function())
        if not np.isfinite(value):
            errors[name] = "Metric returned an undefined/non-finite value."
            return float("nan")
        if caught:
            errors[name] = "; ".join(sorted({str(w.message) for w in caught}))
        return value
    except (ValueError, IndexError, TypeError, ZeroDivisionError, FloatingPointError) as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"
        return float("nan")


def label_metrics(actual, predicted, names, errors):
    unknown = set(names) - LABEL_METRICS
    if unknown:
        raise ValueError(f"Unsupported label metrics: {unknown}")
    from .evaluation import classification_metrics_label as module
    result = {}
    both_classes = len(np.unique(actual)) == 2
    for name in names:
        if name == "add":
            result[name] = add(actual, predicted)
        elif not both_classes and name in {"precision", "recall", "f_score"}:
            function = {"precision": skmetrics.precision_score, "recall": skmetrics.recall_score,
                        "f_score": skmetrics.f1_score}[name]
            result[name] = float(function(actual, predicted, zero_division=0))
            errors[name] = "Single-class input: use sklearn's explicit positive-class definition."
        elif not actual.any() and (name.startswith("affiliation") or name in {"rprecision", "rrecall", "rf", "precision_at_k"}):
            result[name] = float("nan")
            errors[name] = "No ground-truth anomaly event."
        else:
            result[name] = _safe_metric(name, lambda n=name: getattr(module, n)(actual.copy(), predicted.copy()), errors)
    return result


def score_metrics(actual, scores, names, errors):
    unknown = set(names) - SCORE_METRICS
    if unknown:
        raise ValueError(f"Unsupported score metrics (test-label threshold scans are excluded): {unknown}")
    if len(np.unique(actual)) != 2:
        for name in names:
            errors[name] = "Ranking metric undefined for a single ground-truth class."
        return {name: float("nan") for name in names}
    from .evaluation import classification_metrics_score as module
    result = {}
    shared = {}

    def calculate(name):
        # Reuse the shared expensive calculation for each metric family.
        if name in {"R_AUC_ROC", "R_AUC_PR"}:
            if "range" not in shared:
                window = int(np.median(module.get_list_anomaly(actual)))
                shared["range"] = module.metricor_grader.RangeAUC(labels=actual.copy(), score=scores.copy(),
                                                                   window=window, plot_ROC=True)
            return shared["range"][0 if name == "R_AUC_ROC" else 1]
        if name in {"VUS_ROC", "VUS_PR"}:
            if "vus" not in shared:
                window = int(np.median(module.get_list_anomaly(actual)))
                shared["vus"] = module.generate_curve(actual.copy(), scores.copy(), 2 * window)
            return shared["vus"][6 if name == "VUS_ROC" else 7]
        return getattr(module, name)(actual.copy(), scores.copy())

    for name in names:
        result[name] = _safe_metric(name, lambda n=name: calculate(n), errors)
    return result


def evaluate(actual, scores, valid, threshold, cfg):
    actual, scores, valid = np.asarray(actual), np.asarray(scores, np.float64), np.asarray(valid, bool)
    if actual.ndim != 1 or actual.shape != scores.shape or scores.shape != valid.shape:
        raise ValueError("Scores, readiness and labels must preserve the same complete time axis.")
    if not np.isin(actual, [0, 1]).all() or not len(actual):
        raise ValueError("Expected a nonempty binary ground-truth time axis.")
    if not np.isfinite(scores[valid]).all() or not np.isfinite(threshold):
        raise ValueError("Ready scores and threshold must be finite.")
    predicted = (valid & (scores > threshold)).astype(np.int8)
    errors = {}
    labels = label_metrics(actual, predicted, cfg["label_metrics"], errors)
    supplementary = label_metrics(actual, predicted, cfg.get("supplementary_label_metrics", []), errors)
    if valid.any():
        floor = np.nextafter(scores[valid].min(), -np.inf)
        full_scores = np.where(valid, scores, floor)
        ranking = score_metrics(actual, full_scores, cfg["score_metrics"], errors)
        ready_errors = {}
        ready = score_metrics(actual[valid], scores[valid], ["auc_roc", "auc_pr"], ready_errors)
        errors.update({"ready_" + k: v for k, v in ready_errors.items()})
    else:
        ranking = {n: float("nan") for n in cfg["score_metrics"]}
        ready = {n: float("nan") for n in ("auc_roc", "auc_pr")}
        errors["scores"] = "No scorable test point."
    return {"label": labels, "score": ranking, "ready_score": ready, "supplementary": supplementary,
            "event": event_metrics(actual, predicted, valid, cfg["recovery_confirmation"]),
            "undefined_or_warnings": errors,
            "score_convention": "Full time axis; unavailable scores rank below all finite scores. Ready AP/AUC use observed ready points only.",
            "point_adjustment_in_primary_metrics": False}, predicted


def regression_metrics(actual, predicted, names=("mae", "mse", "rmse"), **kwargs):
    """Optional forecasting diagnostics; these are not anomaly-detection metrics."""
    from .evaluation import regression_metrics as module
    if set(names) - set(module.__all__):
        raise ValueError("Unknown regression metric.")
    errors = {}
    result = {n: _safe_metric(n, lambda k=n: getattr(module, k)(actual, predicted, **kwargs), errors) for n in names}
    for n in names:
        if n.startswith("mase") and kwargs.get("seasonality", 2) == 2:
            result[n] = float("nan")
            errors[n] = "Default seasonality=2 returns the sentinel -1; supply an explicit different seasonality."
    return result, errors
