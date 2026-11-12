"""Frozen normal tail ranks; no test-distribution fitting."""
from __future__ import annotations

import numpy as np


class ParallelCalibration:
    def __init__(self, short, dynamic, weights=(.5, .5), alpha=.01, ranking="empirical"):
        if ranking not in {"empirical", "continuous_tail"}:
            raise ValueError("Unknown ranking score mapping.")
        self.ranking = ranking
        self.weights = np.asarray(weights, np.float64)
        if self.weights.shape != (2,) or not np.isfinite(self.weights).all() or (self.weights < 0).any() or not np.isclose(self.weights.sum(), 1):
            raise ValueError("Parallel calibration requires nonnegative fixed weights summing to one.")
        if not 0 < float(alpha) < 1:
            raise ValueError("alpha must lie in (0,1).")
        self.alpha = float(alpha)
        self.references = []
        for values, weight in zip((short, dynamic), self.weights):
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, np.float64)
            if values.ndim != 1 or (weight > 0 and not len(values)) or not np.isfinite(values).all():
                raise ValueError("Each calibration reference must contain finite normal scores.")
            values = np.sort(values).copy()
            values.setflags(write=False)
            self.references.append(values)
        self.tail_parameters = []
        for reference in self.references:
            if not len(reference):
                self.tail_parameters.append(None)
                continue
            anchor = float(reference[int(.95 * (len(reference) - 1))])
            excess = reference[reference > anchor] - anchor
            spread = float(np.quantile(reference, .75) - np.quantile(reference, .25))
            scale = max(float(excess.mean()) if len(excess) else 0., .1 * spread, 1e-3)
            probability = (1 + len(reference) - np.searchsorted(reference, anchor, side="left")) / (len(reference) + 1)
            self.tail_parameters.append(dict(anchor=anchor, scale=scale, probability=float(probability)))

    def transform(self, short, dynamic, short_valid=None, dynamic_valid=None, alpha=None):
        arrays = np.broadcast_arrays(np.asarray(short, np.float64), np.asarray(dynamic, np.float64))
        masks = [np.isfinite(a) if m is None else np.broadcast_to(np.asarray(m, bool), a.shape)
                 for a, m in zip(arrays, (short_valid, dynamic_valid))]
        ranks = []
        for values, valid, reference in zip(arrays, masks, self.references):
            if not np.isfinite(values[valid]).all():
                raise FloatingPointError("Non-finite scorable calibration input.")
            # side='left' counts ties as >= and retains conservative ranks.
            count = len(reference) - np.searchsorted(reference, np.where(valid, values, 0), side="left")
            ranks.append(np.where(valid, (1. + count) / (len(reference) + 1.), 1.))
        adjusted = [p / w if w > 0 else np.full_like(p, np.inf) for p, w in zip(ranks, self.weights)]
        combined = np.minimum(1., np.minimum(*adjusted))
        valid = (masks[0] & (self.weights[0] > 0)) | (masks[1] & (self.weights[1] > 0))
        operating_alpha = self.alpha if alpha is None else float(alpha)
        if not 0 < operating_alpha < 1:
            raise ValueError("alpha must lie in (0,1).")
        scores = -np.log(combined)
        if self.ranking == "continuous_tail":
            evidence = []
            for values, mask, p, weight, parameters in zip(arrays, masks, ranks, self.weights, self.tail_parameters):
                branch = -np.log(p)
                if parameters is not None:
                    above = values > parameters["anchor"]
                    extended = -np.log(parameters["probability"]) + (values - parameters["anchor"]) / parameters["scale"]
                    branch = np.where(above, extended, branch)
                evidence.append(np.where(mask & (weight > 0), branch + np.log(weight) if weight > 0 else -np.inf, -np.inf))
            scores = np.maximum(0., np.maximum(*evidence))
        return dict(scores=np.where(valid, scores, np.nan), valid=valid,
                    p_short=ranks[0], p_dynamic=ranks[1], p_mix=combined,
                    labels=valid & (combined <= operating_alpha))

    def resolution(self, alpha=None):
        alpha = self.alpha if alpha is None else float(alpha)
        return {name: dict(samples=len(ref), minimum_combined_p=(1. / ((len(ref) + 1) * weight) if weight else None),
                          supported=bool(weight and 1. / ((len(ref) + 1) * weight) <= alpha))
                for name, ref, weight in zip(("short", "dynamic"), self.references, self.weights)}

    def as_dict(self):
        return dict(short=self.references[0], dynamic=self.references[1], weights=self.weights.tolist(), alpha=self.alpha,
                    method="parallel_normal", ranking=self.ranking, tail_parameters=self.tail_parameters,
                    score_role="ranking only; alarms always use empirical p_mix",
                    guarantee="pointwise only under valid marginal tail ranks; not anytime-valid; continuous ranking is not a p-value")

    @classmethod
    def from_dict(cls, record):
        return cls(record["short"], record["dynamic"], record["weights"], record["alpha"], record.get("ranking", "empirical"))
