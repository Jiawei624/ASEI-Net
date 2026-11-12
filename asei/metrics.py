from __future__ import annotations

import warnings

import numpy as np

from .tasks import event_ranges
from .config import TAB_ANOMALY_RATIOS

METRICS_PROTOCOL = "detection22_score_topk_v2"
TAB_METRICS_PROTOCOL = "detection22_tab_ratio_v1"
CONTINUOUS_TOPK_PROTOCOLS = {METRICS_PROTOCOL, TAB_METRICS_PROTOCOL}
TAB_THRESHOLD_REFERENCE = {
    "repository": "https://github.com/decisionintelligence/TAB",
    "commit": "5a7e6f80c405eed7c4dd1390f69fbba2fd8ddcf2",
    "path": "ts_benchmark/baselines/self_impl/Anomaly_trans/AnomalyTransformer.py",
}
LABEL_METRIC_ORDER = ("accuracy", "precision", "recall", "f_score", "adjust_accuracy", "adjust_precision",
                      "adjust_recall", "adjust_f_score", "rprecision", "rrecall", "rf", "precision_at_k",
                      "affiliation_precision", "affiliation_recall", "affiliation_f", "add")
SCORE_METRIC_ORDER = ("auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR")
EVENT_METRIC_ORDER = ("events", "detected_events", "event_detection_rate", "event_coverage",
    "event_latter_half_coverage", "recovery_delay_observed_mean", "recovery_observed_events",
    "recovery_censored_fraction", "recovery_delay_detected_mean", "recovery_observed_detected_events",
    "recovery_detected_censored_fraction", "recovery_confirmation", "normal_false_positive_rate",
    "ready_normal_false_positive_rate", "scorable_fraction", "anomaly_scorable_fraction")

LABEL_METRICS = {"accuracy", "precision", "recall", "f_score", "adjust_accuracy", "adjust_precision",
                 "adjust_recall", "adjust_f_score", "rprecision", "rrecall", "rf", "precision_at_k",
                 "affiliation_precision", "affiliation_recall", "affiliation_f", "add"}
SCORE_METRICS = {"auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"}
POINT_ADJUSTED_METRICS = {"adjust_accuracy", "adjust_precision", "adjust_recall", "adjust_f_score"}
REGRESSION_METRICS = {"mae", "mse", "rmse", "mape", "smape", "mase", "wape", "msmape",
                      "mae_norm", "mse_norm", "rmse_norm", "mape_norm", "smape_norm", "mase_norm",
                      "wape_norm", "msmape_norm"}


def point_adjusted_predictions(actual, predicted):
    actual, predicted = _binary_pair(actual, predicted)
    from .evaluation.classification_metrics_label import adjust_predicts
    return adjust_predicts(actual, predicted)


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


def anomaly_ratios(values):
    """TAB ratios are percentages: 1 means 1%, not a fraction of one."""
    if np.isscalar(values):
        values = [values]
    if not isinstance(values, (list, tuple, np.ndarray)) or not len(values):
        raise ValueError("anomaly_ratios must be a nonempty list of percentages.")
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("Anomaly ratios must be numeric percentages.")
    ratios = [float(value) for value in values]
    if any(not np.isfinite(value) or not 0 < value < 100 for value in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("Anomaly ratios must be distinct finite percentages in (0,100).")
    return ratios


def thresholds_from_anomaly_ratios(train_scores, test_scores, ratios=TAB_ANOMALY_RATIOS, prediction_ratio=1.):
    ratios = anomaly_ratios(ratios)
    if prediction_ratio not in ratios:
        raise ValueError("prediction_ratio must be one of anomaly_ratios.")
    train, test = np.asarray(train_scores, np.float64), np.asarray(test_scores, np.float64)
    if train.ndim != 1 or test.ndim != 1 or np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("Threshold inputs must be one-dimensional scores without infinity.")
    # Warmup and unobserved targets remain unavailable, rather than adding zeros.
    train, test = train[np.isfinite(train)], test[np.isfinite(test)]
    if not len(train):
        raise ValueError("TAB thresholds require finite training scores.")
    combined = np.concatenate([train, test])
    thresholds = np.percentile(combined, 100 - np.asarray(ratios))
    entries = [{"anomaly_ratio": ratio, "threshold": float(value)} for ratio, value in zip(ratios, thresholds)]
    return {"method": "tab_anomaly_ratio", "threshold_source": "normal_fit_and_unlabeled_test",
        "anomaly_ratios": ratios, "prediction_ratio": float(prediction_ratio), "thresholds": entries,
        "threshold": float(thresholds[ratios.index(prediction_ratio)]), "comparison": ">",
        "train_scores": len(train), "test_scores": len(test), "combined_scores": len(combined),
        "reference": dict(TAB_THRESHOLD_REFERENCE)}


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
    if valid.shape != actual.shape:
        raise ValueError("Readiness must preserve the event time axis.")
    if isinstance(confirmation, (bool, np.bool_)) or not isinstance(confirmation, (int, np.integer)) or confirmation < 1:
        raise ValueError("Recovery confirmation must be a positive integer.")
    events = event_ranges(actual)
    coverage, detected, later_coverage, recovery, censored = [], [], [], [], 0
    detected_recovery, detected_censored = [], 0
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
                if detected[-1]:
                    detected_recovery.append(int(positions[0]))
                found = True
        if not found:
            censored += 1
            detected_censored += int(detected[-1])
    normal = actual == 0
    ready_normal = normal & valid
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    return {"events": len(events), "detected_events": int(sum(detected)),
            "event_detection_rate": mean(detected), "event_coverage": mean(coverage),
            "event_latter_half_coverage": mean(later_coverage), "add": add(actual, predicted),
            "recovery_delay_observed_mean": mean(recovery), "recovery_observed_events": len(recovery),
            "recovery_censored_fraction": censored / len(events) if events else float("nan"),
            "recovery_delay_detected_mean": mean(detected_recovery),
            "recovery_observed_detected_events": len(detected_recovery),
            "recovery_detected_censored_fraction": detected_censored / sum(detected) if any(detected) else float("nan"),
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


def precision_at_k(actual, scores):
    """Top-K score precision; boundary ties receive their expected contribution."""
    actual, scores = np.asarray(actual), np.asarray(scores, np.float64)
    if actual.ndim != 1 or actual.shape != scores.shape or not len(actual):
        raise ValueError("Precision at K requires equally sized, nonempty vectors.")
    if not np.isin(actual, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Precision at K requires binary labels and finite scores.")
    k = int(actual.sum())
    if not k:
        return float("nan")
    boundary = np.partition(scores, len(scores) - k)[len(scores) - k]
    above, tied = scores > boundary, scores == boundary
    remaining = k - int(above.sum())
    return float((actual[above].sum() + remaining * actual[tied].mean()) / k)


def label_metrics(actual, predicted, names, errors, scores=None):
    unknown = set(names) - LABEL_METRICS
    if unknown:
        raise ValueError(f"Unsupported label metrics: {unknown}")
    from .evaluation import classification_metrics_label as module
    result = {}
    actual, predicted = _binary_pair(actual, predicted)
    adjusted = None
    shared = {}
    range_index = {"rrecall": 4, "rprecision": 7, "rf": 8}

    def native_value(name):
        if name == "precision_at_k":
            return precision_at_k(actual, predicted if scores is None else scores)
        if name in range_index:
            if "range" not in shared:
                shared["range"] = module.metricor_grader.metric_new(actual.copy(), predicted.copy(), plot_ROC=False)
            return shared["range"][range_index[name]]
        return getattr(module, name)(actual.copy(), predicted.copy())
    for name in names:
        if name == "add":
            result[name] = _safe_metric(name, lambda: add(actual, predicted), errors)
        elif name in POINT_ADJUSTED_METRICS:
            if adjusted is None:
                adjusted = point_adjusted_predictions(actual, predicted)
            base = name[len("adjust_"):]
            result[name] = _safe_metric(name, lambda n=base: getattr(module, n)(actual, adjusted), errors)
        elif not actual.any() and (name.startswith("affiliation") or name in {"rprecision", "rrecall", "rf", "precision_at_k"}):
            result[name] = float("nan")
            errors[name] = "No ground-truth anomaly event."
        else:
            result[name] = _safe_metric(name, lambda n=name: native_value(n), errors)
    return result


def score_metrics(actual, scores, names, errors, workers=1):
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
                if workers > 1:
                    shared["vus"] = module.generate_curve(actual.copy(), scores.copy(), 2 * window, workers=workers)
                else:
                    shared["vus"] = module.generate_curve(actual.copy(), scores.copy(), 2 * window)
            return shared["vus"][6 if name == "VUS_ROC" else 7]
        return getattr(module, name)(actual.copy(), scores.copy())

    for name in names:
        result[name] = _safe_metric(name, lambda n=name: calculate(n), errors)
    return result


def _evaluation_scores(actual, scores, valid):
    actual, scores, valid = np.asarray(actual), np.asarray(scores, np.float64), np.asarray(valid, bool)
    if actual.ndim != 1 or actual.shape != scores.shape or scores.shape != valid.shape:
        raise ValueError("Scores, readiness and labels must preserve the same complete time axis.")
    if not np.isin(actual, [0, 1]).all() or not len(actual):
        raise ValueError("Expected a nonempty binary ground-truth time axis.")
    if not np.isfinite(scores[valid]).all():
        raise ValueError("Ready scores must be finite.")
    full_scores = np.where(valid, scores, np.nextafter(scores[valid].min(), -np.inf)) if valid.any() else np.zeros(len(actual))
    return actual, scores, valid, full_scores


def evaluate(actual, scores, valid, threshold, cfg):
    actual, scores, valid, full_scores = _evaluation_scores(actual, scores, valid)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite.")
    predicted = (valid & (scores > threshold)).astype(np.int8)
    errors = {}
    if valid.any():
        ranking = score_metrics(actual, full_scores, cfg["score_metrics"], errors)
    else:
        full_scores = np.zeros(len(actual), np.float64)
        ranking = {n: float("nan") for n in cfg["score_metrics"]}
        errors["scores"] = "No scorable test point."
    labels = label_metrics(actual, predicted, cfg["label_metrics"], errors, full_scores)
    if not valid.any() and "precision_at_k" in labels:
        labels["precision_at_k"] = float("nan")
        errors["precision_at_k"] = "No scorable test point."
    return {"label": labels, "score": ranking,
            "undefined_or_warnings": errors,
            "protocol": METRICS_PROTOCOL,
            "score_convention": "Full time axis; unavailable scores rank below all finite scores. Precision at K uses continuous scores with fractional boundary ties.",
            "point_adjustment_in_primary_metrics": bool(POINT_ADJUSTED_METRICS.intersection(labels)),
            "label_convention": "Only adjust_* metrics use point-adjusted alarms; precision_at_k ranks continuous scores; all other label metrics and ADD use original alarms."}, predicted


def evaluate_anomaly_ratios(actual, scores, valid, calibration, cfg):
    """One continuous-score evaluation and one label evaluation per preset ratio."""
    actual, scores, valid, full_scores = _evaluation_scores(actual, scores, valid)
    score_errors = {}
    if valid.any():
        ranking = score_metrics(actual, full_scores, cfg["score_metrics"], score_errors)
    else:
        ranking = {name: float("nan") for name in cfg["score_metrics"]}
        score_errors["scores"] = "No scorable test point."
    rows, alarms = [], []
    for entry in calibration["thresholds"]:
        predicted = (valid & (scores > entry["threshold"])).astype(np.int8)
        errors = {}
        values = label_metrics(actual, predicted, cfg["label_metrics"], errors, full_scores)
        if not valid.any() and "precision_at_k" in values:
            values["precision_at_k"] = float("nan")
            errors["precision_at_k"] = "No scorable test point."
        rows.append({**entry, "metrics": values, "undefined_or_warnings": errors,
                     "test_alarm_fraction": float(predicted[valid].mean()) if valid.any() else float("nan")})
        alarms.append(predicted)
    return {"label_by_ratio": rows, "score": ranking, "undefined_or_warnings": score_errors,
        "protocol": TAB_METRICS_PROTOCOL, "score_protocol": METRICS_PROTOCOL, "anomaly_ratio_unit": "percent",
        "threshold_source": calibration["threshold_source"],
        "score_convention": "Continuous scores on the full time axis; unavailable points rank below finite scores.",
        "label_convention": "Each preset ratio is evaluated separately; only adjust_* uses point adjustment; no best ratio is selected."}, np.stack(alarms)


def regression_metrics(actual, predicted, names=("mae", "mse", "rmse"), **kwargs):
    """Forecasting metrics on observed values; inputs retain their channel schema."""
    from .evaluation import regression_metrics as module
    if set(names) - set(module.__all__):
        raise ValueError("Unknown regression metric.")
    actual, predicted = np.asarray(actual, np.float64), np.asarray(predicted, np.float64)
    if actual.shape != predicted.shape or actual.ndim not in (1, 2) or not actual.size:
        raise ValueError("Regression metrics require nonempty, equally shaped one- or two-dimensional arrays.")
    errors = {}
    result = {n: _safe_metric(n, lambda k=n: getattr(module, k)(actual, predicted, **kwargs), errors) for n in names}
    return result, errors
