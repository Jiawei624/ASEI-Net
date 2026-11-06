from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Record


TASK_SUPERVISION_PROTOCOL = "causal_event_prefix_v2"


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
                  families=("level", "variance", "temporal", "relation")) -> TaskPair:
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
        variable_fallback = attempt >= 80 and family in {"temporal", "relation"}
        gap_options = [g for g in (4, 16, 64) if g + layout * min_event + suffix + prefix <= max_span]
        gap = int(rng.choice(gap_options)) if layout == 2 else 0
        budget = (max_span - suffix - prefix - gap) // layout
        choices = sorted({l for l in (1, 4, 16, 64, window, 2 * window, 4 * window)
                          if min_event <= l <= budget and (family != "relation" or l <= len(donor.z))})
        if not choices:
            raise ValueError("Normal record too short for configured task layout.")
        # Short complete spans remain usable when missing observations prevent
        # longer events. Four points is already in the configured duration grid.
        lengths = ([min_event] * layout if variable_fallback and attempt >= 120
                   else [int(rng.choice(choices)) for _ in range(layout)])
        span = prefix + sum(lengths) + gap + suffix
        target_origin = int(rng.integers(target_begin, len(source.z) - span + 1))
        origin, stop = target_origin - window, target_origin + span
        clean = source.z[origin:stop].copy()
        masks = source.mask[origin:stop].copy()
        corrupt, labels = clean.copy(), np.zeros(len(clean), np.int8)
        start, recipes, failed = window + prefix, [], False
        for length in lengths:
            end = start + length
            eligible = np.flatnonzero(masks[start:end].all(0))
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
                            "subtype": subtype, "donor_start": donor_start})
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
        return TaskPair(Record(clean, masks.copy(), window), Record(corrupt, masks, window), labels, recipe)
    raise ValueError(f"Cannot construct a changed {family} task from observed normal data "
                     f"(seed={seed}, index={index}, attempts=160); this task needs observed, "
                     "locally varying channels and, for relation tasks, a usable normal donor.")
