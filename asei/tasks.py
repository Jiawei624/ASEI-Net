from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Record


TASK_SUPERVISION_PROTOCOL = "causal_event_prefix_v2"


class TaskGenerationError(ValueError):
    """A sampled task layout could not produce an observed change."""


def validate_pair_observability(pair, active=None, tolerance=1e-6):
    """Validate the final stored observations, after every intervention and fill.

    A nonzero floating-point proposal is insufficient: blending and assignment
    into float32 may erase it. Never hide such a task by dropping its positives.
    """
    clean, corrupt = pair.clean, pair.corrupt
    previous = np.roll(corrupt.mask, 1, axis=0)
    previous[0] = False
    valid = corrupt.mask & previous
    valid &= (np.arange(len(pair.labels)) >= corrupt.target_start)[:, None]
    if active is not None:
        valid &= active
    changed = clean.mask & corrupt.mask & (np.abs(corrupt.z.astype(np.float64) - clean.z) > tolerance)
    events = event_ranges(pair.labels)
    if not events:
        raise TaskGenerationError("Task has no injected event.")
    for start, stop in events:
        observed = np.maximum.accumulate(changed[start:stop], axis=0) & valid[start:stop]
        if not observed.any():
            raise TaskGenerationError(
                f"Task index={pair.recipe.get('index')} family={pair.recipe.get('family')} "
                f"event=[{start},{stop}) has no observable, scorable change after final storage.")


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


def phase_weights(labels, clean_valid, corrupt_valid, positive_valid=None, onset_fraction=.5):
    """Weights over a pair sum to one; positives and negatives each sum to 1/2."""
    labels = np.asarray(labels, bool)
    if not 0 < onset_fraction < 1:
        raise ValueError("onset_fraction must be in (0,1).")
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
        phases = [(p[eligible[p]], w) for p, w in zip(phases, (onset_fraction, 1 - onset_fraction)) if len(p)]
        phases = [(p, w) for p, w in phases if len(p)]
        if not phases:
            raise ValueError("No scorable positive sample for an injected event.")
        total_weight = sum(w for _, w in phases)
        for p, weight in phases:
            positive[p] = 0.5 / len(events) * weight / total_weight / len(p)
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


def channel_phase_weights(clean, corrupt, labels, clean_valid, corrupt_valid, onset_fraction=.5):
    """Use actual paired changes to supervise every affected channel, not just argmax."""
    if not 0 < onset_fraction < 1:
        raise ValueError("onset_fraction must be in (0,1).")
    labels = np.asarray(labels, bool)
    events = event_ranges(labels)
    channels = list(range(clean.z.shape[1]))
    positive = observable_channel_mask(clean, corrupt, labels,
        [dict(start=start, stop=stop, channels=channels) for start, stop in events])
    clean_valid, corrupt_valid = np.asarray(clean_valid, bool), np.asarray(corrupt_valid, bool)
    if clean_valid.shape != clean.z.shape or corrupt_valid.shape != clean.z.shape:
        raise ValueError("Channel supervision masks must preserve [time, channel].")
    cw = np.zeros_like(clean.z, dtype=np.float32)
    aw = np.zeros_like(cw)
    affected = np.zeros_like(positive)
    for start, stop in events:
        affected[start:stop] = positive[start:stop].any(0)
        eligible = positive[start:stop] & corrupt_valid[start:stop]
        observed_channels = np.flatnonzero(eligible.any(0))
        if not len(observed_channels):
            raise ValueError("No observable affected channel for a paired event.")
        for channel in observed_channels:
            positions = np.flatnonzero(eligible[:, channel]) + start
            onset = positions[0]
            first, interior = positions[positions < onset + 8], positions[positions >= onset + 8]
            phases = [(ix, weight) for ix, weight in ((first, onset_fraction), (interior, 1 - onset_fraction)) if len(ix)]
            normalization = sum(weight for _, weight in phases)
            for ix, weight in phases:
                aw[ix, channel] = .5 * weight / normalization / len(events) / len(observed_channels) / len(ix)
    negative_clean = clean_valid.astype(np.float64)
    # The unchanged prefix of an affected channel is unlabelled. Unaffected
    # channels remain negative even during another channel's injected event.
    negative_corrupt = (corrupt_valid & ~affected).astype(np.float64)
    if events:
        negative_corrupt[events[0][1]:] *= 2
    total = negative_clean.sum() + negative_corrupt.sum()
    if total <= 0:
        raise ValueError("Channel supervision has no usable normal location.")
    cw[:] = .5 * negative_clean / total
    aw += (.5 * negative_corrupt / total).astype(np.float32)
    return positive, cw, aw


def generate_reference_pair(source, donor, window, index, seed, families, policy,
                            hard_anchors=None, hard_fraction=.25, donor_candidates=16, boundary_blend=4,
                            relation_matcher=None):
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
            if relation_matcher is not None:
                cost = relation_matcher(clean, masks, start, stop, channels, donor, left)
                if cost < best:
                    best, selected, donor_start = float(cost), values.astype(clean.dtype), left
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
        try:
            validate_pair_observability(pair)
        except TaskGenerationError:
            continue
        return pair
    raise TaskGenerationError(f"Cannot construct a support-preserving task (seed={seed}, index={index}).")


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
                # Test the exact values that will be stored, rather than the raw
                # delta. A short blended event or float32 rounding can erase an
                # otherwise nonzero proposal; keep its valid matched donor then.
                proposed = (pair.clean.z[start:stop, i] + strength * delta).astype(values.dtype)
                observed = pair.clean.mask[start:stop, i] & pair.corrupt.mask[start:stop, i]
                changed = np.abs(proposed.astype(np.float64) - pair.clean.z[start:stop, i]) > 1e-6
                if not (changed & observed).any():
                    fallbacks.append(dict(channel=i, reason="lag_intervention_has_no_observable_effect_after_storage"))
                    continue
                values[start:stop, i] = proposed
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
