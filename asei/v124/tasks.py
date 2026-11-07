"""Typed normal anchors and lag interventions preserving fitted normal residuals."""
from __future__ import annotations

import numpy as np

from ..data import Record


def normal_difficulty(record, supports):
    seen = record.mask & np.roll(record.mask, 1, axis=0)
    seen[0] = False
    change = np.abs(np.diff(record.z.astype(np.float64), axis=0, prepend=record.z[:1]))
    rarity = np.zeros_like(change)
    for c, support in enumerate(supports):
        if support is None:
            continue
        change[:, c] = record.z[:, c] != np.roll(record.z[:, c], 1)
        states, counts = np.unique(record.z[record.mask[:, c], c], return_counts=True)
        if len(states):
            ix = np.searchsorted(states, record.z[:, c]).clip(max=len(states) - 1)
            rarity[:, c] = -np.log(counts[ix] / counts.sum()) * record.mask[:, c]
    return (np.minimum(change, 10) * seen + rarity).sum(-1) / seen.sum(-1).clip(min=1)


def typed_donor_matcher(supports):
    def cost(clean, masks, start, stop, channels, donor, left):
        length = stop - start
        total, count = 0., 0
        affected = set(int(i) for i in channels)
        for i, support in enumerate(supports):
            if not masks[start:stop, i].all() or not donor.mask[left:left + length, i].all():
                continue
            a = clean[start:stop, i].astype(np.float64)
            b = donor.z[left:left + length, i].astype(np.float64)
            if support is not None:
                # State probabilities and symbol disagreement are invariant to renumbering.
                states = np.unique(np.r_[a, b])
                distance = sum((np.mean(a == s) - np.mean(b == s)) ** 2 for s in states)
                if i in affected:
                    distance += .25 * np.mean(a != b)
            else:
                ma, mb, sa, sb = a.mean(), b.mean(), a.std(), b.std()
                distance = (ma - mb) ** 2 + (sa - sb) ** 2
                if i in affected:
                    distance += .25 * np.mean(np.square((a - ma) / max(sa, .001) - (b - mb) / max(sb, .001)))
            total, count = total + distance, count + 1
        return total / max(1, count)
    return cost


class ResidualLagTasks:
    """Sparse normal VAR(1); perturb one cross-lag contribution, retain the residual.

    This constructs a statistical dependence intervention, not a physical causal model.
    Categorical channels continue to use support-preserving matched donors.
    """
    def __init__(self, fit, supports, relation, seed, max_peers=4, max_points=10000):
        self.edges = {}
        numeric = np.array([s is None for s in supports])
        times = np.arange(1, len(fit.z))
        if len(times) > max_points:
            times = np.sort(np.random.default_rng(seed).choice(times, max_points, replace=False))
        for i in np.flatnonzero(numeric):
            candidates = np.flatnonzero(numeric & (np.arange(len(supports)) != i))
            peers = candidates[np.argsort(-np.abs(relation[i, candidates]), kind="stable")[:max_peers]]
            if not len(peers):
                continue
            columns = np.r_[i, peers]
            usable = fit.mask[times, i] & fit.mask[times[:, None] - 1, columns].all(-1)
            ix = times[usable]
            if len(ix) < max(32, 4 * (len(columns) + 1)):
                continue
            x = np.c_[np.ones(len(ix)), fit.z[ix[:, None] - 1, columns]].astype(np.float64)
            y = fit.z[ix, i].astype(np.float64)
            penalty = np.eye(x.shape[1]) * (.001 * len(ix))
            penalty[0, 0] = 0
            beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            variance = np.std(x[:, 2:], axis=0) * np.abs(beta[2:])
            order = np.argsort(-variance, kind="stable")
            self.edges[int(i)] = [(int(peers[j]), float(beta[j + 2])) for j in order if variance[j] > 1e-4]

    def apply(self, pair, seed):
        if pair.recipe["family"] != "relation":
            return pair
        rng = np.random.default_rng(np.random.SeedSequence([seed, pair.recipe["index"], 124]))
        values = pair.corrupt.z.copy()
        for event in pair.recipe["events"]:
            start, stop = event["start"], event["stop"]
            interventions, fallbacks = [], []
            for i in event["channels"]:
                eligible = [(j, b) for j, b in self.edges.get(i, []) if j not in event["channels"]]
                if not eligible:
                    fallbacks.append(dict(channel=i, reason="no_usable_continuous_external_edge"))
                    continue
                j, beta = eligible[int(rng.integers(len(eligible)))]
                lag = int(rng.choice([l for l in (1, 4, 16) if l < start]))
                t = np.arange(start, stop)
                good = pair.clean.mask[t - 1, j] & pair.clean.mask[t - 1 - lag, j]
                if not good.all():
                    fallbacks.append(dict(channel=i, reason="missing_lag_context"))
                    continue
                delta = beta * (pair.clean.z[t - 1 - lag, j] - pair.clean.z[t - 1, j])
                if not np.isfinite(delta).all() or np.max(np.abs(delta)) <= 1e-6:
                    fallbacks.append(dict(channel=i, reason="lag_intervention_has_no_observable_effect"))
                    continue
                # Smooth the intervention strength at the splice; the original residual stays additive.
                blend = min(4, (stop - start) // 4)
                strength = np.ones(stop - start, np.float32)
                if blend:
                    strength[:blend] = np.arange(1, blend + 1) / (blend + 1)
                    strength[-blend:] = strength[:blend][::-1]
                values[start:stop, i] = pair.clean.z[start:stop, i] + strength * delta
                interventions.append(dict(channel=i, source=j, coefficient=beta, added_lag=lag,
                                          delta_rms=float(np.sqrt(np.mean(np.square(strength * delta))))))
            event["residual_lag_interventions"] = interventions
            event["donor_fallbacks"] = fallbacks
            event["subtype"] = "residual_lag_with_recorded_donor_fallback" if interventions else "matched_donor"
        previous = values[0].copy()
        for t in range(len(values)):
            previous = np.where(pair.corrupt.mask[t], values[t], previous)
            values[t] = previous
        pair.corrupt = Record(values, pair.corrupt.mask, pair.corrupt.target_start)
        return pair
