"""Observed-target forecasting diagnostics for the two frozen prediction branches."""
from __future__ import annotations

import numpy as np

from .metrics import REGRESSION_METRICS, regression_metrics
from .resources import cooperate


class _ColumnScaler:
    def __init__(self, mean, scale):
        self.mean, self.scale = float(mean), float(scale)

    def transform(self, values):
        return (np.asarray(values, np.float64) - self.mean) / self.scale


def evaluate_forecasts(sequence, training, scaler, names, test_start=0, seasonality=1, channel_names=None):
    """Pool point errors over observed targets; retain real gaps in MASE history."""
    if set(names) - REGRESSION_METRICS:
        raise ValueError("Unknown regression metric.")
    if not 0 <= test_start <= len(sequence):
        raise ValueError("Invalid test start for forecast evaluation.")
    channels = sequence.z.shape[1]
    channel_names = list(channel_names) if channel_names is not None else list(range(channels))
    if len(channel_names) != channels or training.z.shape[1] != channels:
        raise ValueError("Forecast, history and channel schemas differ.")
    ready = np.arange(test_start, len(sequence)) >= sequence.window
    results = {}
    details = {"seasonality": seasonality, "per_channel": [], "coverage": {}, "undefined_or_warnings": {},
               "scope": "Observed test targets after prediction warmup, including anomalous test points.",
               "scaler": "Fixed normal-fit mean and scale; raw values are recovered from stored FP32 standardized inputs.",
               "history": "Normal fit prefix only; missing history stays missing and is never concatenated across gaps.",
               "aggregation": "Point-weighted means; pooled RMSE and WAPE. MASE averages only channels with a defined positive training-naive scale."}
    for branch, branch_index in (("self", 0), ("relation", 1)):
        if branch == "relation" and channels == 1:
            continue
        entries = {name: [] for name in names}
        raw_numerator = raw_denominator = normalized_numerator = normalized_denominator = 0.0
        observed_total = channels_with_targets = 0
        for channel in range(channels):
            cooperate()
            observed = sequence.mask[test_start:, channel] & ready
            count = int(observed.sum())
            if not count:
                details["per_channel"].append({"branch": branch, "channel": channel_names[channel],
                                               "observed_targets": 0, "reason": "No observed target after warmup."})
                continue
            channels_with_targets += 1
            observed_total += count
            column_scaler = _ColumnScaler(scaler.mean[channel], scaler.scale[channel])
            mean, scale = column_scaler.mean, column_scaler.scale
            actual = (sequence.z[test_start:, channel][observed].astype(np.float64) * scale + mean)[:, None]
            predicted = (sequence.prediction[test_start:, channel, branch_index][observed].astype(np.float64) * scale + mean)[:, None]
            history = training.z[:, channel].astype(np.float64) * scale + mean
            history[~training.mask[:, channel]] = np.nan
            values, errors = regression_metrics(actual, predicted, names, scaler=column_scaler,
                                                 hist_data=history[:, None], seasonality=seasonality)
            details["per_channel"].append({"branch": branch, "channel": channel_names[channel],
                "observed_targets": count, "metrics": values, "undefined_or_warnings": errors})
            for name in names:
                entries[name].append((values[name], count))
            raw_numerator += float(np.abs(actual - predicted).sum())
            raw_denominator += float(np.abs(actual).sum())
            normalized_actual, normalized_predicted = column_scaler.transform(actual), column_scaler.transform(predicted)
            normalized_numerator += float(np.abs(normalized_actual - normalized_predicted).sum())
            normalized_denominator += float(np.abs(normalized_actual).sum())
        pooled, coverage = {}, {}
        for name in names:
            available = entries[name]
            finite = [(value, count) for value, count in available if np.isfinite(value)]
            metric_count = sum(count for _, count in finite)
            coverage[name] = {"channels": len(finite), "targets": metric_count}
            if name in {"wape", "wape_norm"}:
                numerator, denominator = ((raw_numerator, raw_denominator) if name == "wape" else
                                           (normalized_numerator, normalized_denominator))
                value = 100 * numerator / denominator if denominator > 0 else np.nan
                coverage[name] = {"channels": channels_with_targets, "targets": observed_total}
            elif not finite or (len(finite) != len(available) and name not in {"mase", "mase_norm"}):
                value = np.nan
            elif name in {"rmse", "rmse_norm"}:
                value = np.sqrt(sum(value ** 2 * count for value, count in finite) / metric_count)
            else:
                value = sum(value * count for value, count in finite) / metric_count
            pooled[name] = float(value) if np.isfinite(value) else float("nan")
            if not np.isfinite(value):
                details["undefined_or_warnings"][branch + "." + name] = (
                    "Metric is undefined on the observed test targets or normal training scale; see per-channel reasons.")
        results[branch] = pooled
        details["coverage"][branch] = coverage
    return results, details
