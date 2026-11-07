"""Normal-only preparation and paired TBPTT with a fixed normal backbone."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..compat import load_torch_checkpoint
from ..config import fingerprint, write_json
from ..data import Record
from ..tasks import (NormalTaskPolicy, TaskGenerationError, generate_reference_pair,
                     observable_positive_mask, phase_weights)
from ..resources import cooperate
from .features import forecast_errors
from .cache import PairCache124
from .tasks import normal_difficulty, typed_donor_matcher


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def portable(item):
        # Keep weights_only=True compatible with both NumPy 1.x and 2.x environments.
        if isinstance(item, np.ndarray):
            return torch.from_numpy(item.copy())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {k: portable(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [portable(v) for v in item]
        return item
    torch.save(portable(value), temporary)
    temporary.replace(path)


def _macro(loss, valid):
    counts = valid.sum(1)
    mean = (loss * valid).sum(1) / counts.clamp(min=1)
    return (mean * (counts > 0)).sum(-1) / (counts > 0).sum(-1).clamp(min=1)


def fit_forecaster(normal, fit, selection, cfg, directory, key, seed):
    path = Path(directory) / "representation.pt"
    if path.exists():
        saved = load_torch_checkpoint(path, map_location=normal.schema.amp_mean.device)
        if saved["key"] != key:
            raise ValueError("Normal checkpoint identity differs; start a new run.")
        normal.load_state_dict(saved["model"])
        if not saved.get("memory_complete", False):
            normal.memory.fit(fit, seed, cfg["memory124"]["pool_size"], cfg["memory124"]["rare_fraction"])
            saved.update(model=normal.state_dict(), memory_complete=True)
            _save(path, saved)
        return saved.get("history", [])
    predictor, w = normal.predictor, normal.predictor.window
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    settings = cfg["stage1"]
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=settings["lr"], weight_decay=settings.get("weight_decay", 1e-4))
    batch = int(settings["batch_size"])
    limit = int(settings.get("max_train_points", 0))
    selection_limit = int(settings.get("max_validation_points", 4096))
    rng = np.random.default_rng(seed)
    def epoch(record, training):
        predictor.train(training)
        times = np.arange(max(w, record.target_start), len(record.z), max(1, int(settings.get("stride", 1))))
        if training:
            rng.shuffle(times)
            if limit:
                times = times[:limit]
        elif selection_limit and len(times) > selection_limit:
            times = times[np.linspace(0, len(times) - 1, selection_limit).astype(int)]
        sums = np.zeros(record.z.shape[1])
        counts = np.zeros_like(sums)
        for begin in range(0, len(times), batch):
            cooperate()
            ix = times[begin:begin + batch]
            history_index = ix[:, None] + np.arange(-w, 0)
            device = normal.schema.amp_mean.device
            x = torch.as_tensor(record.z[history_index], device=device)
            mask = torch.as_tensor(record.mask[history_index], device=device)
            target = torch.as_tensor(record.z[ix], device=device)
            observed = torch.as_tensor(record.mask[ix], device=device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                predicted, logits, available = predictor(x, mask)
                loss = F.huber_loss(predicted, target[..., None].expand_as(predicted), reduction="none")
                symbols = normal.schema.ids(target)
                ce = -F.log_softmax(logits, -1).gather(-1, symbols[..., None, None].expand(-1, -1, 2, 1)).squeeze(-1)
                loss = torch.where(normal.schema.categorical[None, :, None], ce, loss)
                valid = available & observed[..., None]
                channel_count = valid.sum((0, 2))
                channel_loss = (loss * valid).sum((0, 2)) / channel_count.clamp(min=1)
                value = channel_loss[channel_count > 0].mean() if valid.any() else loss.new_zeros(())
                value = value + cfg["stage2"].get("relation_l1", 1e-3) * predictor.weights().abs().mean()
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite normal forecasting loss.")
                if training and valid.any():
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1)
                    optimizer.step()
                    predictor.project()
            sums += (loss.detach() * valid).sum((0, 2)).cpu().numpy()
            counts += valid.sum((0, 2)).cpu().numpy()
        if not (counts > 0).any():
            raise ValueError("No usable forecasting targets in the normal split.")
        return float((sums[counts > 0] / counts[counts > 0]).mean())
    history, best, stale = [], float("inf"), 0
    for number in range(settings["epochs"]):
        training_loss, validation_loss = epoch(fit, True), epoch(selection, False)
        history.append(dict(epoch=number + 1, train_loss=training_loss, validation_loss=validation_loss))
        print(f"v124 forecast {number + 1}: train={training_loss:.6f} selection={validation_loss:.6f}", flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            _save(path, dict(key=key, model=normal.state_dict(), history=history))
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    normal.load_state_dict(load_torch_checkpoint(path, map_location=normal.schema.amp_mean.device)["model"])
    normal.predictor.eval()
    # Dictionaries contain real fitting windows and are fixed before teacher learning.
    normal.memory.fit(fit, seed, cfg["memory124"]["pool_size"], cfg["memory124"]["rare_fraction"])
    _save(path, dict(key=key, model=normal.state_dict(), history=history, memory_complete=True))
    return history


def _sequence_starts(record, w, length, count, seed):
    available = len(record.z) - max(w, record.target_start)
    length = min(length, available - 1)
    if length < 2:
        raise ValueError("Normal selection/teacher segment is too short.")
    targets = np.random.default_rng(seed).integers(max(w, record.target_start), len(record.z) - length, size=count)
    # Consume real quiet history to initialize states, but score only target positions.
    return [(int(t - w), int(w + length)) for t in targets]


def fit_teacher(model, fit, selection, cfg, directory, key, seed):
    path = Path(directory) / "teacher.pt"
    normal, teacher = model.normal, model.normal.teacher
    if path.exists():
        saved = load_torch_checkpoint(path, map_location=normal.schema.amp_mean.device)
        if saved["key"] != key:
            raise ValueError("Teacher checkpoint identity differs; start a new run.")
        normal.load_state_dict(saved["model"])
        return saved.get("history", [])
    for p in teacher.parameters():
        p.requires_grad_(True)
    settings, w = cfg["teacher"], normal.predictor.window
    optimizer = torch.optim.Adam(teacher.parameters(), lr=settings["lr"])
    block_size = min(int(cfg["stage2"]["block_length"]), int(settings.get("block_length", 64)))
    history, best, stale = [], float("inf"), 0
    def epoch(record, sequences, training):
        teacher.train(training)
        total = 0.
        for first in range(0, len(sequences), settings["batch_sequences"]):
            batch = sequences[first:first + settings["batch_sequences"]]
            starts, lengths = zip(*batch)
            direct, closed = None, None
            if training:
                optimizer.zero_grad(set_to_none=True)
            accumulated, has_grad = 0., False
            # Fixed per-sequence/channel denominators avoid block-size dependent targets.
            denominator = torch.zeros(len(batch), normal.schema.channels, device=normal.schema.amp_mean.device)
            for offset in range(0, max(lengths), block_size):
                length = min(block_size, max(lengths) - offset)
                f = normal.schema.block([record] * len(batch), [s + offset for s in starts], length)
                following = normal.schema.block([record] * len(batch), [s + offset + 1 for s in starts], length)
                if offset == 0:
                    f["sv"][:, 0] = False  # Same zero-state boundary as a standalone paired task.
                f["target"] &= (torch.arange(offset, offset + length, device=f["v"].device)[None, :, None] >= w)
                good = f["sv"] & f["lv"] & following["lv"] & f["target"]
                denominator += good.sum(1)
            for offset in range(0, max(lengths), block_size):
                cooperate()
                length = min(block_size, max(lengths) - offset)
                f = normal.schema.block([record] * len(batch), [s + offset for s in starts], length)
                following = normal.schema.block([record] * len(batch), [s + offset + 1 for s in starts], length)
                if offset == 0:
                    f["sv"][:, 0] = False
                f["target"] &= (torch.arange(offset, offset + length, device=f["v"].device)[None, :, None] >= w)
                valid = f["sv"] & f["lv"] & following["lv"] & f["target"]
                with torch.set_grad_enabled(training):
                    hidden, direct = model.teacher_block(f, direct, False)
                    reference_hidden, closed = model.teacher_block(f, closed, True)
                    now, future = teacher.losses(hidden, f, following)
                    rn, rf = teacher.losses(reference_hidden, f, following)
                    channel_loss = ((now + future + rn + rf) * .25 * valid).sum(1) / denominator.clamp(min=1)
                    loss = (channel_loss.sum(-1) / (denominator > 0).sum(-1).clamp(min=1)).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite normal state-teacher loss.")
                    if training and valid.any():
                        loss.backward()
                        has_grad = True
                direct, closed = direct.detach(), closed.detach()
                accumulated += float(loss.detach())
            if training and has_grad:
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1)
                optimizer.step()
                teacher.cell.project()
            total += accumulated * len(batch)
        return total / len(sequences)
    validation = _sequence_starts(selection, w, settings["sequence_length"], settings["validation_sequences"], 7101)
    for number in range(settings["epochs"]):
        sequences = _sequence_starts(fit, w, settings["sequence_length"], settings["train_sequences"], seed + 7200 + number)
        training_loss, validation_loss = epoch(fit, sequences, True), epoch(selection, validation, False)
        history.append(dict(epoch=number + 1, train_loss=training_loss, validation_loss=validation_loss))
        print(f"v124 teacher {number + 1}: train={training_loss:.6f} selection={validation_loss:.6f}", flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            _save(path, dict(key=key, model=normal.state_dict(), history=history))
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    normal.load_state_dict(load_torch_checkpoint(path, map_location=normal.schema.amp_mean.device)["model"])
    normal.freeze()
    return history


@torch.no_grad()
def fit_energy_scales(model, fit, cfg):
    """Complete fitting replay, only block-local descriptors and scalar accumulators."""
    normal, teacher = model.normal, model.normal.teacher
    sums = torch.zeros_like(normal.memory.energy_scale, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    state = model.initial_state(1).normal
    length = cfg["stage2"]["block_length"]
    for start in range(0, len(fit.z), length):
        f = normal.schema.block([fit], [start], min(length, len(fit.z) - start))
        long = normal.memory.long_evidence(f)
        lavail = torch.stack([long["mv"], long["jv"]], -1)
        sums[:, 1] += (long["long_dist"] * lavail).double().sum((0, 1))
        counts[:, 1] += lavail.sum((0, 1))
        for t in range(f["v"].shape[1]):
            step = {k: v[:, t] for k, v in f.items()}
            prior, logits = teacher.prior(state)
            ref, pair, available, external = normal.memory.reference(step, prior, logits,
                cfg["stage2"]["reference_trim"], cfg["stage2"]["reference_prior"],
                external=model.variant != "no_external")
            distance, sv = normal.memory.short_evidence(step, ref, pair, available, external)
            sums[:, 0] += (distance * sv).double().sum(0)
            counts[:, 0] += sv.sum(0)
            valid = step["sv"] & available
            state = teacher.cell(teacher.adapt(ref, pair).reshape(-1, 32), state.reshape(-1, 32)).reshape_as(state)
            state.masked_fill_(~valid[..., None], 0)
        cooperate()
    normal.memory.energy_scale.copy_((sums / counts.clamp(min=1)).clamp(min=1e-3).float())
    return dict(counts=counts.cpu().tolist(), scales=normal.memory.energy_scale.cpu().tolist())


def build_pairs(model, source, donor, cfg, directory, split, seed, names, relation_tasks=None):
    settings, normal = cfg["stage2"], model.normal
    count = settings["train_pairs"] if split == "train" else settings["validation_pairs"]
    key = fingerprint(dict(split=split, seed=seed, stage2=settings, shape=source.z.shape,
                           normal_key=cfg["_normal_key"], teacher_key=cfg["_teacher_key"]))
    cache = PairCache124(Path(directory) / ("tasks124_" + split), key, normal.schema.channels,
                         cfg["runtime"].get("task_cache_memory_mb", 256))
    if len(cache) >= count:
        return cache
    policy = NormalTaskPolicy(source, settings["discrete_max_values"], cfg["data"].get("channel_types"), names)
    policy.supports = [None if s is None else np.asarray(s, np.float32) for s in normal.metadata["supports"]]
    difficulty = normal_difficulty(source, policy.supports)
    candidates = np.arange(max(source.target_start, normal.predictor.window), len(source.z))
    hard = candidates[difficulty[candidates] >= np.quantile(difficulty[candidates], .95)]
    matcher = typed_donor_matcher(policy.supports)
    # A normal source is forecast once. Task slices share its exact errors after
    # their W-point warm-up; perturbed windows are always recomputed. Memory maps
    # avoid duplicating an entire long normal record in RAM for this optimization.
    error_path, ready_path = cache.directory / "source_error.npy", cache.directory / "source_ready.npy"
    marker = cache.directory / "source_forecast.json"
    complete = marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("key") == key
    if not complete or not error_path.exists() or not ready_path.exists():
        shape = (*source.z.shape, 2)
        source_error = np.lib.format.open_memmap(error_path, mode="w+", dtype=np.float32, shape=shape)
        source_ready = np.lib.format.open_memmap(ready_path, mode="w+", dtype=bool, shape=shape)
        try:
            forecast_errors(source, normal, cfg["runtime"]["feature_batch_size"], output=(source_error, source_ready))
            source_error.flush()
            source_ready.flush()
        finally:
            source_error._mmap.close()
            source_ready._mmap.close()
        write_json(marker, dict(key=key, shape=shape))
    source_error, source_ready = np.load(error_path, mmap_mode="r"), np.load(ready_path, mmap_mode="r")
    try:
        for index in range(len(cache), count):
            pair = generate_reference_pair(source, donor, normal.predictor.window, index, seed,
                settings["families"], policy, hard, settings["hard_negative_fraction"],
                settings["donor_candidates"], settings["boundary_blend"], relation_matcher=matcher)
            if relation_tasks is not None:
                pair = relation_tasks.apply(pair, seed)
            begin, end = pair.recipe["source_start"], pair.recipe["source_stop"]
            clean_error, clean_ready = source_error[begin:end].copy(), source_ready[begin:end].copy()
            clean_error[:normal.predictor.window] = 0
            clean_ready[:normal.predictor.window] = False
            corrupt_error, corrupt_ready, reused = forecast_errors(pair.corrupt, normal, cfg["runtime"]["feature_batch_size"],
                pair.clean, clean_error, clean_ready)
            pair.recipe["forecast_rows_reused"] = reused
            pair.recipe["generator124"] = settings["relation_generator"]
            cache.append(pair, (clean_error, corrupt_error), (clean_ready, corrupt_ready))
            if index == 0 or (index + 1) % 32 == 0:
                print(f"v124 {split} tasks {index + 1}/{count}", flush=True)
    finally:
        source_error._mmap.close()
        source_ready._mmap.close()
    return cache


def _recovery(clean, corrupt, labels):
    same = clean.mask & corrupt.mask & (np.abs(clean.z - corrupt.z) <= 1e-6)
    same[1:] &= same[:-1].copy()
    same[0] = False
    started = np.maximum.accumulate(np.asarray(labels, bool))
    age = np.zeros(clean.z.shape[1], np.float32)
    result = np.zeros_like(clean.z)
    for t in range(len(labels)):
        age = np.where(same[t] & ~labels[t] & started[t], age + 1, 0)
        result[t] = np.minimum(age / 8, 1)
    return result


def fit_detector(model, training, validation, cfg, directory, key, seed):
    path = Path(directory) / "detector.pt"
    if path.exists():
        saved = load_torch_checkpoint(path, map_location=model.normal.schema.amp_mean.device)
        if saved["key"] != key:
            raise ValueError("Detector checkpoint identity differs; use a new run.")
        model.load_state_dict(saved["model"])
        return saved.get("history", [])
    settings = cfg["stage2"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=settings["lr"])
    rng = np.random.default_rng(seed)
    def epoch(cache, learn):
        model.train(learn)
        order = rng.permutation(len(cache)) if learn else np.arange(len(cache))
        totals = {name: 0. for name in ("loss", "classification", "state", "tail")}
        updates = 0
        for begin in range(0, len(order), settings["batch_pairs"]):
            cooperate()
            batch = [cache.get(int(i)) for i in order[begin:begin + settings["batch_pairs"]]]
            clean, corrupt, labels, recipes = zip(*[(p[0], p[1], p[2], p[3]) for p in batch])
            records = [v[0] for v in clean] + [v[0] for v in corrupt]
            errors = [v[1] for v in clean] + [v[1] for v in corrupt]
            ready = [v[2] for v in clean] + [v[2] for v in corrupt]
            size = len(clean)
            weights, recovery, state_denominators, recovery_denominators = [], [], [], []
            for left, right, y in zip(clean, corrupt, labels):
                active = model.normal.memory.m_active.any(-1).cpu().numpy()
                def usable(record):
                    previous = np.roll(record.mask, 1, axis=0)
                    previous[0] = False
                    return (record.mask & previous & active).any(-1) & (np.arange(len(y)) >= record.target_start)
                cvalid, avalid = usable(left[0]), usable(right[0])
                positive = observable_positive_mask(left[0], right[0], y)
                cw, aw = phase_weights(y, cvalid, avalid, positive)
                weights.append((cw, aw))
                ramp = _recovery(left[0], right[0], y)
                recovery.append(ramp)
                def channel_usable(record):
                    previous = np.roll(record.mask, 1, axis=0)
                    previous[0] = False
                    return record.mask & previous & active & (np.arange(len(y)) >= record.target_start)[:, None]
                state_denominators.append(max(1, channel_usable(left[0]).sum()))
                recovery_denominators.append(max(1, (ramp * channel_usable(right[0])).sum()))
            max_length = max(len(r.z) for r in records)
            weight_arrays = [w[0] for w in weights] + [w[1] for w in weights]
            device = model.normal.schema.amp_mean.device
            whole_weight = torch.zeros(2 * size, max_length, device=device)
            whole_truth = torch.zeros_like(whole_weight)
            for i, wa in enumerate(weight_arrays):
                whole_weight[i, :len(wa)] = torch.as_tensor(wa, device=device)
            for i, y in enumerate(labels):
                whole_truth[size + i, :len(y)] = torch.as_tensor(y, device=device)
            tail = whole_weight.new_zeros(())
            tail_coefficient = whole_weight.new_zeros(())
            tail_selected = torch.zeros_like(whole_weight, dtype=torch.bool)
            tail_count = 1
            if model.variant not in {"no_tail", "energy_only"} and settings["lambda_tail"]:
                # A detached first pass fixes the whole-task tail. The second pass applies
                # its exact scalar gradient through each TBPTT block, without a T*C*H graph.
                snapshot = torch.zeros_like(whole_weight)
                snapshot_valid = torch.zeros_like(tail_selected)
                snapshot_state = None
                with torch.no_grad():
                    for start in range(0, max_length, settings["block_length"]):
                        length = min(settings["block_length"], max_length - start)
                        f = model.normal.schema.block(records, [start] * len(records), length, errors, ready)
                        out, snapshot_state = model.forward_block(f, snapshot_state)
                        snapshot[:, start:start + length] = out["dynamic"]
                        snapshot_valid[:, start:start + length] = out["dynamic_valid"]
                    negative = (whole_truth == 0) & (whole_weight > 0) & snapshot_valid
                    positive = (whole_truth > 0) & (whole_weight > 0) & snapshot_valid
                    if negative.any() and positive.any():
                        positions = negative.flatten().nonzero(as_tuple=False).flatten()
                        tail_count = max(1, int(np.ceil(settings["tail_fraction"] * len(positions))))
                        selected = positions[snapshot.flatten()[positions].topk(tail_count).indices]
                        tail_selected.flatten()[selected] = True
                        tail = snapshot.flatten()[selected].mean()
                        tail_coefficient = 2 * (torch.sigmoid(settings["tail_margin"] + tail - snapshot)
                                                * whole_weight * positive).sum() / size
            state, teacher_state = None, None
            if learn:
                optimizer.zero_grad(set_to_none=True)
            pending = False
            for start in range(0, max_length, settings["block_length"]):
                length = min(settings["block_length"], max_length - start)
                f = model.normal.schema.block(records, [start] * len(records), length, errors, ready)
                with torch.no_grad():
                    normal_f = {k: v[:size] for k, v in f.items()}
                    targets, teacher_state = model.teacher_block(normal_f, teacher_state, False)
                with torch.set_grad_enabled(learn):
                    out, state = model.forward_block(f, state, return_aux=True)
                    device = out["dynamic"].device
                    weight = torch.zeros(2 * size, length, device=device)
                    truth = torch.zeros_like(weight)
                    ramp = torch.zeros(size, length, model.normal.schema.channels, device=device)
                    for i, (wa, record) in enumerate(zip(weight_arrays, records)):
                        stop = min(start + length, len(record.z))
                        if stop > start:
                            weight[i, :stop - start] = torch.as_tensor(wa[start:stop], device=device)
                    for i, y in enumerate(labels):
                        stop = min(start + length, len(y))
                        if stop > start:
                            truth[size + i, :stop - start] = torch.as_tensor(y[start:stop], device=device)
                            ramp[i, :stop - start] = torch.as_tensor(recovery[i][start:stop], device=device)
                    valid = out["dynamic_valid"]
                    classification = (F.binary_cross_entropy_with_logits(out["dynamic"], truth, reduction="none") * weight * valid).sum() / size
                    eligible = out["reference_valid"] & f["target"]
                    clean_difference = F.huber_loss(out["observed"][:size], targets, reduction="none").mean(-1)
                    rec_difference = F.huber_loss(out["observed"][size:], targets, reduction="none").mean(-1)
                    # Whole-task denominators, not an average of block averages.
                    denom = torch.as_tensor(state_denominators, device=device)
                    clean_state = (clean_difference * eligible[:size]).sum((1, 2)) / denom
                    rec_denom = torch.as_tensor(recovery_denominators, device=device)
                    recovered = (rec_difference * ramp * eligible[size:]).sum((1, 2)) / rec_denom
                    state_loss = (clean_state + recovered).mean()
                    positives = (truth > 0) & (weight > 0) & valid
                    tail_loss = classification.new_zeros(())
                    if tail_selected.any():
                        tail_loss = 2 * (F.softplus(settings["tail_margin"] + tail - out["dynamic"]) * weight * positives).sum() / size
                        selected = tail_selected[:, start:start + length]
                        tail_loss = tail_loss + tail_coefficient * ((out["dynamic"] - out["dynamic"].detach()) * selected).sum() / tail_count
                    loss = classification
                    if model.variant not in {"no_state", "mlp", "energy_only"}:
                        loss = loss + settings["lambda_state"] * state_loss
                    if model.variant != "no_tail":
                        loss = loss + settings["lambda_tail"] * tail_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite paired detector objective.")
                    if learn and (weight * valid).sum() > 0:
                        loss.backward()
                        pending = True
                    for name, value in zip(totals, (loss, classification, state_loss, tail_loss)):
                        totals[name] += float(value.detach()) * size
                state = state.detach()
            if learn and pending:
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                updates += 1
        return {**{k: v / len(cache) for k, v in totals.items()}, "optimizer_updates": updates}
    history, best, stale = [], float("inf"), 0
    for number in range(settings["epochs"]):
        train, valid = epoch(training, True), epoch(validation, False)
        history.append(dict(epoch=number + 1, train=train, selection=valid))
        print(f"v124 detector {number + 1}: train={train['loss']:.6f} selection={valid['loss']:.6f}", flush=True)
        if valid["loss"] < best:
            best, stale = valid["loss"], 0
            _save(path, dict(key=key, model=model.state_dict(), history=history))
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(load_torch_checkpoint(path, map_location=model.normal.schema.amp_mean.device)["model"])
    model.eval()
    return history
