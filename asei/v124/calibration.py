"""Frozen normal tail ranks; no test-distribution fitting."""
from __future__ import annotations

import numpy as np


class ParallelCalibration:
    def __init__(self, short, dynamic, weights=(.5, .5), alpha=.01):
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
        return dict(scores=np.where(valid, -np.log(combined), np.nan), valid=valid,
                    p_short=ranks[0], p_dynamic=ranks[1], p_mix=combined,
                    labels=valid & (combined <= operating_alpha))

    def resolution(self, alpha=None):
        alpha = self.alpha if alpha is None else float(alpha)
        return {name: dict(samples=len(ref), minimum_combined_p=(1. / ((len(ref) + 1) * weight) if weight else None),
                          supported=bool(weight and 1. / ((len(ref) + 1) * weight) <= alpha))
                for name, ref, weight in zip(("short", "dynamic"), self.references, self.weights)}

    def as_dict(self):
        return dict(short=self.references[0], dynamic=self.references[1], weights=self.weights.tolist(), alpha=self.alpha,
                    method="parallel_normal", guarantee="pointwise only under valid marginal tail ranks; not anytime-valid")

    @classmethod
    def from_dict(cls, record):
        return cls(record["short"], record["dynamic"], record["weights"], record["alpha"])
