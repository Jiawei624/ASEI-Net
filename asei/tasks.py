from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Record


TASK_SUPERVISION_PROTOCOL = "causal_event_prefix_v2"


class TaskGenerationError(ValueError):
    """A sampled task layout could not produce an observed change."""


def event_ranges(labels):
    values = np.asarray(labels, dtype=bool)
    edges = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def observable_positive_mask(clean: Record, corrupt: Record, labels, tolerance=1e-6):
    """Supervise an event only after its first observed change from the clean pair.

    Event boundaries remain intact for delay/coverage evaluation. Later unchanged
    samples stay positive because their observed history can reveal the event.
    """
    labels = np.asarray(labels, bool)
    if clean.z.shape != corrupt.z.shape or clean.mask.shape != clean.z.shape or corrupt.mask.shape != clean.z.shape:
        raise ValueError("Paired records must have the same observation shape.")
    if labels.shape != (len(clean.z),):
        raise ValueError("Event labels must preserve the paired time axis.")
    changed = np.any(clean.mask & corrupt.mask &
                     (np.abs(corrupt.z.astype(np.float64) - clean.z) > tolerance), axis=1)
    usable = np.zeros(len(labels), bool)
    for start, stop in event_ranges(labels):
        usable[start:stop] = np.maximum.accumulate(changed[start:stop])
    return usable


def phase_weights(labels, clean_valid, corrupt_valid, positive_valid=None):
    """Weights over a pair sum to one; positives and negatives each sum to 1/2."""
    labels = np.asarray(labels, bool)
    clean_valid, corrupt_valid = np.asarray(clean_valid, bool), np.asarray(corrupt_valid, bool)
    if labels.ndim != 1 or clean_valid.shape != labels.shape or corrupt_valid.shape != labels.shape:
        raise ValueError("Labels and scoring masks must preserve the same one-dimensional time axis.")
    eligible = corrupt_valid.copy()
    if positive_valid is not None:
        positive_valid = np.asarray(positive_valid, bool)
        if positive_valid.shape != labels.shape:
            raise ValueError("Positive supervision mask must preserve the event time axis.")
        eligible &= positive_valid
    events = event_ranges(labels)
    positive = np.zeros(len(labels), np.float64)
    for start, stop in events:
        observed = np.flatnonzero(eligible[start:stop])
        if not len(observed):
            raise ValueError("No scorable positive sample for an injected event.")
        onset = start + int(observed[0]) if positive_valid is not None else start
        cut = min(onset + 8, stop)
        phases = [np.arange(onset, cut), np.arange(cut, stop)]
        phases = [p[eligible[p]] for p in phases if len(p)]
        phases = [p for p in phases if len(p)]
        if not phases:
            raise ValueError("No scorable positive sample for an injected event.")
        for p in phases:
            positive[p] = 0.5 / len(events) / len(phases) / len(p)
    if not events:
        raise ValueError("A task pair requires a scorable event.")
    clean_negative = np.asarray(clean_valid, np.float64)
    corrupt_negative = (~labels & corrupt_valid).astype(np.float64)
    if events:
        corrupt_negative[events[0][1]:] *= 2
    negative_total = clean_negative.sum() + corrupt_negative.sum()
    if not negative_total:
        raise ValueError("Task pair has no scorable negative sample.")
    return (0.5 * clean_negative / negative_total).astype(np.float32), (
        positive + 0.5 * corrupt_negative / negative_total).astype(np.float32)


@dataclass
class TaskPair:
    clean: Record
    corrupt: Record
    labels: np.ndarray
    recipe: dict


def generate_pair(source: Record, donor: Record, window: int, index: int, seed: int,
                  families=("level", "variance", "temporal", "relation"),
                  hard_anchors=None, hard_fraction=0., relation_sampler=None, allowed_channels=None) -> TaskPair:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    c = source.z.shape[1]
    families = [f for f in families if c > 1 or f != "relation"]
    if not families or set(families) - {"level", "variance", "temporal", "relation"}:
        raise ValueError("No valid task family.")
    family = families[index % len(families)]
    layout = 1 + ((index // len(families)) % 2)
    target_begin = max(window, source.target_start)
    available = len(source.z) - target_begin
    min_event = 4 if family in {"temporal", "relation"} else 1
    min_total = layout * min_event + (4 if layout == 2 else 0) + 1
    if available < min_total:
        raise ValueError(f"Normal source has {available} targets; {family}/{layout} events need at least {min_total}.")
    max_span = min(available, 128 + layout * 4 * window + 64 + window + 128)
    suffix = min(window + 128, max(1, (max_span - min_total) // 2))
    prefix = min(128, max(0, (max_span - suffix - min_total) // 3))
    # Keep the original draws for tasks that already succeed, including cached
    # prefixes. Sparse actuator changes need a second pass over variable channels.
    for attempt in range(160):
        variable_fallback = attempt >= 80 and (family == "temporal" or (family == "relation" and relation_sampler is None))
        gap_options = [g for g in (4, 16, 64) if g + layout * min_event + suffix + prefix <= max_span]
        gap = int(rng.choice(gap_options)) if layout == 2 else 0
        budget = (max_span - suffix - prefix - gap) // layout
        choices = sorted({l for l in (1, 4, 16, 64, window, 2 * window, 4 * window)
                          if min_event <= l <= budget and (family != "relation" or l <= len(donor.z))})
        if not choices:
            raise ValueError("Normal record too short for configured task layout.")
        # Short complete spans remain usable when missing observations prevent
        # longer events. Four points is already in the configured duration grid.
        lengths = ([min_event] * layout if attempt >= 120 and (variable_fallback or relation_sampler is not None)
                   else [int(rng.choice(choices)) for _ in range(layout)])
        span = prefix + sum(lengths) + gap + suffix
        target_origin = int(rng.integers(target_begin, len(source.z) - span + 1))
        hard_sample = hard_anchors is not None and len(hard_anchors) and rng.random() < hard_fraction
        if hard_sample:
            target_origin = int(np.clip(int(rng.choice(hard_anchors)) - prefix,
                                        target_begin, len(source.z) - span))
        origin, stop = target_origin - window, target_origin + span
        clean = source.z[origin:stop].copy()
        masks = source.mask[origin:stop].copy()
        corrupt, labels = clean.copy(), np.zeros(len(clean), np.int8)
        start, recipes, failed = window + prefix, [], False
        for length in lengths:
            end = start + length
            eligible = np.flatnonzero(masks[start:end].all(0))
            if allowed_channels is not None:
                eligible = eligible[np.asarray(allowed_channels, bool)[eligible]]
            if variable_fallback and len(eligible):
                local = clean[start:end, eligible]
                eligible = eligible[(np.ptp(local, axis=0) > 1e-6)
                                    & (local.std(axis=0, dtype=np.float64) > 1e-6)]
            if len(eligible) == 0:
                failed = True
                break
            counts = sorted({1, int(np.ceil(0.1 * c)), int(np.ceil(0.5 * c)), c})
            counts = [n for n in counts if n <= len(eligible) and (family != "relation" or n < c)]
            if not counts:
                failed = True
                break
            affected = rng.choice(eligible, size=int(rng.choice(counts)), replace=False)
            reference = clean[start:end, affected].copy()
            donor_start = None
            metadata = {}
            if family == "level":
                magnitude = np.exp(rng.uniform(np.log(0.2), np.log(3), len(affected)))
                magnitude *= rng.choice([-1, 1], len(affected))
                subtype = "shift" if rng.random() < 0.5 else "drift"
                ramp = np.ones(length) if subtype == "shift" else np.arange(1, length + 1) / length
                replacement = reference + ramp[:, None] * magnitude
            elif family == "variance":
                subtype = "noise" if rng.random() < 0.5 else "stagnation"
                if subtype == "stagnation":
                    replacement = np.broadcast_to(reference[0], reference.shape).copy()
                else:
                    replacement = reference + rng.normal(0, rng.uniform(0.2, 2), reference.shape)
            elif family == "temporal":
                subtype = "shift" if rng.random() < 0.5 else "reorder"
                permutation = np.roll(np.arange(length), int(rng.integers(1, length))) if subtype == "shift" else rng.permutation(length)
                replacement = reference[permutation]
            elif relation_sampler is not None:
                sampled = relation_sampler(clean, masks, start, end, affected, rng)
                if sampled is None:
                    failed = True
                    break
                replacement, metadata = sampled
                subtype, donor_start = "normal_donor", metadata["donor_start"]
            else:
                subtype = "matched_donor"
                replacement = reference.copy()
                for _ in range(30):
                    donor_start = int(rng.integers(0, len(donor.z) - length + 1))
                    if donor.mask[donor_start:donor_start + length, affected].all():
                        shape = donor.z[donor_start:donor_start + length, affected].copy()
                        if variable_fallback:
                            # Compute moments in float64 for almost-flat sensors.
                            shape = shape.astype(np.float64)
                            matched = reference.astype(np.float64)
                        else:
                            matched = reference
                        ds, rs = shape.std(0), matched.std(0)
                        if (ds > 1e-6).all() and (rs > 1e-6).all():
                            replacement = ((shape - shape.mean(0)) / ds * rs + matched.mean(0)).astype(reference.dtype)
                            if not variable_fallback or np.any(np.abs(replacement - reference) > 1e-6):
                                break
            changed = np.any(np.abs(replacement - reference) > 1e-6)
            if not changed:
                failed = True
                break
            corrupt[start:end, affected] = replacement
            labels[start:end] = 1
            recipes.append({"start": start, "stop": end, "channels": affected.tolist(),
                            "subtype": subtype, "donor_start": donor_start, **metadata})
            start = end + gap
        if failed:
            continue
        # Refill from the perturbed observed past. Reusing clean forward-filled values
        # after an event would introduce the paired clean trajectory into model inputs.
        last = clean[0].copy()
        for t in range(len(corrupt)):
            last = np.where(masks[t], corrupt[t], last)
            corrupt[t] = last
        recipe = {"seed": seed, "index": index, "family": family, "layout": layout,
                  "source_start": origin, "source_stop": stop, "prefix": prefix,
                  "recovery_suffix": suffix, "events": recipes, "attempt": attempt}
        if attempt >= 80:
            recipe["fallback"] = "variable_channels" if variable_fallback else "additional_attempts"
        if hard_anchors is not None:
            recipe["hard_normal_source"] = bool(hard_sample)
        return TaskPair(Record(clean, masks.copy(), window), Record(corrupt, masks, window), labels, recipe)
    raise TaskGenerationError(f"Cannot construct a changed {family} task from observed normal data "
                     f"(seed={seed}, index={index}, attempts=160); this task needs observed, "
                     "locally varying channels and, for relation tasks, a usable normal donor.")


REFERENCE_TASK_PROTOCOL = "support_preserving_pairs_v2"


class NormalTaskPolicy:
    """Channel supports come exclusively from the normal fit record."""

    def __init__(self, fit, max_values=16, channel_types=None, channel_names=None):
        channel_types = channel_types or {}
        names = channel_names or [str(i) for i in range(fit.z.shape[1])]
        unknown = set(channel_types) - set(names)
        if unknown:
            raise ValueError("Unknown channel type names: " + ", ".join(sorted(unknown)))
        self.supports = []
        for i, name in enumerate(names):
            values = np.unique(fit.z[fit.mask[:, i], i])
            kind = channel_types.get(name, "auto")
            if kind not in {"auto", "continuous", "discrete"}:
                raise ValueError("Channel type must be auto, continuous or discrete.")
            discrete = kind == "discrete" or (kind == "auto" and len(values) <= max_values)
            self.supports.append(values if discrete else None)

    def hard_anchors(self, source, window):
        difference = np.abs(np.diff(source.z.astype(np.float64), axis=0, prepend=source.z[:1]))
        seen = source.mask & np.roll(source.mask, 1, axis=0)
        seen[0] = False
        difficulty = (np.minimum(difference, 10) * seen).sum(-1) / seen.sum(-1).clip(min=1)
        # Rare normal discrete states also supply difficult, legitimate negatives.
        for c, support in enumerate(self.supports):
            if support is None or len(support) < 2:
                continue
            values, counts = np.unique(source.z[source.mask[:, c], c], return_counts=True)
            if len(values):
                positions = np.searchsorted(values, source.z[:, c]).clip(max=len(values) - 1)
                difficulty += source.mask[:, c] * (-np.log(counts[positions] / counts.sum())) / source.z.shape[1]
        begin = max(window, source.target_start)
        candidates = np.arange(begin, len(source.z))
        if not len(candidates):
            return candidates
        return candidates[difficulty[candidates] >= np.quantile(difficulty[candidates], .95)]


def observable_channel_mask(clean, corrupt, labels, events, tolerance=1e-6):
    changed = clean.mask & corrupt.mask & (np.abs(corrupt.z.astype(np.float64) - clean.z) > tolerance)
    result = np.zeros_like(changed)
    for event in events:
        start, stop = event["start"], event["stop"]
        channels = event.get("channels", [])
        if channels:
            result[start:stop, channels] = np.maximum.accumulate(changed[start:stop, channels], axis=0)
    return result & np.asarray(labels, bool)[:, None]


def generate_reference_pair(source, donor, window, index, seed, families, policy,
                            hard_anchors=None, hard_fraction=.25, donor_candidates=16, boundary_blend=4):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 919]))
    family_count = len([f for f in families if source.z.shape[1] > 1 or f != "relation"])
    if not family_count:
        raise ValueError("No valid task family.")
    allowed = np.array([support is None or len(support) > 1 for support in policy.supports])
    if not allowed.any():
        raise TaskGenerationError("Normal fit record has no changeable channel for support-preserving tasks.")

    def relation_sampler(clean, masks, start, stop, channels, task_rng):
        reference = clean[start:stop, channels].astype(np.float64)
        length = stop - start
        unaffected = np.setdiff1d(np.arange(source.z.shape[1]), channels)
        best, selected, donor_start = float("inf"), None, None
        ref_mean, ref_std = reference.mean(0), reference.std(0)
        ref_shape = (reference - ref_mean) / ref_std.clip(min=1e-3)
        for _ in range(donor_candidates):
            left = int(task_rng.integers(0, len(donor.z) - length + 1))
            if not donor.mask[left:left + length, channels].all():
                continue
            values = donor.z[left:left + length, channels].astype(np.float64)
            if not np.any(np.abs(values - reference) > 1e-6):
                continue
            mean, std = values.mean(0), values.std(0)
            cost = np.square(mean - ref_mean).mean() + np.square(std - ref_std).mean()
            cost += .25 * np.square((values - mean) / std.clip(min=1e-3) - ref_shape).mean()
            if len(unaffected):
                complete = donor.mask[left:left + length, unaffected].all(0) & masks[start:stop, unaffected].all(0)
                if complete.any():
                    dc = donor.z[left:left + length, unaffected][:, complete]
                    sc = clean[start:stop, unaffected][:, complete]
                    cost += np.square(dc.mean(0, dtype=np.float64) - sc.mean(0, dtype=np.float64)).mean()
            if cost < best:
                best, selected, donor_start = float(cost), values.astype(clean.dtype), left
        return None if selected is None else (selected, {"donor_start": donor_start, "donor_match_cost": best})

    for retry in range(64):
        # A multiple of the family/layout cycle preserves the requested task stratum.
        base_index = index + retry * 2 * family_count
        try:
            pair = generate_pair(source, donor, window, base_index, seed, families, hard_anchors, hard_fraction,
                                 relation_sampler=relation_sampler, allowed_channels=allowed)
        except TaskGenerationError:
            continue
        corrupt = pair.clean.z.copy()
        family = pair.recipe["family"]
        failed = False
        for event in pair.recipe["events"]:
            start, stop = event["start"], event["stop"]
            length, channels = stop - start, event["channels"]
            changed_channels = []
            reference = pair.clean.z[start:stop, channels]
            replacement = pair.corrupt.z[start:stop, channels].copy()
            for j, c in enumerate(channels):
                support = policy.supports[c]
                if support is not None:
                    if len(support) < 2:
                        replacement[:, j] = reference[:, j]
                    elif family in {"level", "variance"}:
                        alternatives = support[np.abs(support - reference[0, j]) > 1e-6]
                        state = reference[0, j] if family == "variance" or not len(alternatives) else rng.choice(alternatives)
                        replacement[:, j] = state
                    # Temporal permutations and unscaled donors preserve discrete support.
                elif family == "relation" and boundary_blend:
                    blend = min(int(boundary_blend), length // 4)
                    if blend:
                        ramp = np.arange(1, blend + 1, dtype=np.float32) / (blend + 1)
                        replacement[:blend, j] = reference[:blend, j] + ramp * (replacement[:blend, j] - reference[:blend, j])
                        replacement[-blend:, j] = reference[-blend:, j] + ramp[::-1] * (replacement[-blend:, j] - reference[-blend:, j])
                if np.any(np.abs(replacement[:, j] - reference[:, j]) > 1e-6):
                    changed_channels.append(c)
            if not changed_channels:
                failed = True
                break
            corrupt[start:stop, channels] = replacement
            event["channels"] = changed_channels
            if family == "relation":
                blend = min(int(boundary_blend), length // 4)
                event["interior"] = [start + blend, stop - blend]
        if failed:
            continue
        last = pair.clean.z[0].copy()
        for t in range(len(corrupt)):
            last = np.where(pair.corrupt.mask[t], corrupt[t], last)
            corrupt[t] = last
        pair.corrupt = Record(corrupt, pair.corrupt.mask, window)
        pair.recipe.update(index=index, base_index=base_index, support_retry=retry, task_protocol=REFERENCE_TASK_PROTOCOL)
        return pair
    raise TaskGenerationError(f"Cannot construct a support-preserving task (seed={seed}, index={index}).")
