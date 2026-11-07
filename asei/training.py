"""Normal training, fixed evidence preparation and resumable checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .compat import load_torch_checkpoint
from .config import fingerprint, write_json
from .tasks import NormalTaskPolicy, generate_reference_pair
from .resources import cooperate
from .progress import Progress
from .features import forecast_errors
from .cache import PairCache
from .tasks import normal_difficulty, typed_donor_matcher


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


class StageCheckpoint:
    """Best weights plus resumable last-epoch state; an early best is not completion."""
    def __init__(self, path, module, optimizer, key, rng=None):
        self.path, self.module, self.optimizer = Path(path), module, optimizer
        self.last = self.path.with_name(self.path.stem + "_last.pt")
        self.key, self.rng = key, rng
        self.history, self.best, self.stale, self.complete = [], float("inf"), 0, False
        source = self.last if self.last.exists() else self.path
        if source.exists():
            saved = load_torch_checkpoint(source, map_location=next(module.parameters()).device)
            if saved.get("key") != key:
                raise ValueError("Checkpoint identity differs; start a new run: " + str(source))
            module.load_state_dict(saved["model"])
            self.history = saved.get("history", [])
            self.best, self.stale = saved.get("best", float("inf")), saved.get("stale", 0)
            self.complete = saved.get("complete", False)
            if not self.complete:
                optimizer.load_state_dict(saved["optimizer"])
                torch.set_rng_state(saved["torch_rng"].cpu())
                if torch.cuda.is_available() and "cuda_rng" in saved:
                    torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
                if rng is not None:
                    rng.bit_generator.state = saved["numpy_rng"]
                print(f"resume {self.path.stem} epoch={len(self.history) + 1}", flush=True)

    def record(self, entry, validation):
        self.history.append(entry)
        improved = validation < self.best
        self.best, self.stale = (validation, 0) if improved else (self.best, self.stale + 1)
        saved = dict(key=self.key, model=self.module.state_dict(), optimizer=self.optimizer.state_dict(),
                     history=self.history, best=self.best, stale=self.stale, complete=False,
                     torch_rng=torch.get_rng_state())
        if self.rng is not None:
            saved["numpy_rng"] = self.rng.bit_generator.state
        if torch.cuda.is_available():
            saved["cuda_rng"] = torch.cuda.get_rng_state_all()
        if improved:
            _save(self.path, saved)
        _save(self.last, saved)

    def finish(self):
        saved = load_torch_checkpoint(self.path, map_location=next(self.module.parameters()).device)
        self.module.load_state_dict(saved["model"])
        saved.update(complete=True, history=self.history)
        # Optimizer and random state are only needed while training is incomplete.
        for name in ("optimizer", "torch_rng", "cuda_rng", "numpy_rng"):
            saved.pop(name, None)
        _save(self.path, saved)
        self.last.unlink(missing_ok=True)
        self.complete = True
        return self.history


def fit_forecaster(normal, fit, selection, cfg, directory, key, seed):
    path = Path(directory) / "representation.pt"
    predictor, w = normal.predictor, normal.predictor.window
    for parameter in predictor.parameters():
        parameter.requires_grad_(True)
    settings = cfg["stage1"]
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=settings["lr"], weight_decay=settings.get("weight_decay", 1e-4))
    batch = int(settings["batch_size"])
    limit = int(settings.get("max_train_points", 0))
    selection_limit = int(settings.get("max_validation_points", 4096))
    rng = np.random.default_rng(seed)
    checkpoint = StageCheckpoint(path, normal, optimizer, key, rng)
    interval = cfg["runtime"].get("log_interval_seconds", 30)
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
        progress = Progress(f"stage1 {'train' if training else 'validation'} batches", (len(times) + batch - 1) // batch, interval)
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
            progress.update(begin // batch + 1)
        if not (counts > 0).any():
            raise ValueError("No usable forecasting targets in the normal split.")
        return float((sums[counts > 0] / counts[counts > 0]).mean())
    for number in range(len(checkpoint.history), settings["epochs"]) if not checkpoint.complete else []:
        if checkpoint.stale >= settings["patience"]:
            break
        began = time.perf_counter()
        training_loss, validation_loss = epoch(fit, True), epoch(selection, False)
        checkpoint.record(dict(epoch=number + 1, train_loss=training_loss, validation_loss=validation_loss), validation_loss)
        print(f"stage1 epoch={number + 1} train={training_loss:.6f} validation={validation_loss:.6f} "
              f"seconds={time.perf_counter() - began:.1f}", flush=True)
    history = checkpoint.history if checkpoint.complete else checkpoint.finish()
    normal.freeze()
    # Dictionaries contain real fitting windows and are fixed before teacher learning.
    saved = load_torch_checkpoint(path, map_location=normal.schema.amp_mean.device)
    if not saved.get("memory_complete", False):
        print("memory construction start", flush=True)
        normal.memory.fit(fit, seed, cfg["memory"]["pool_size"], cfg["memory"]["rare_fraction"], interval)
        saved.update(model=normal.state_dict(), memory_complete=True)
        _save(path, saved)
        print("memory construction complete", flush=True)
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
    for p in teacher.parameters():
        p.requires_grad_(True)
    settings, w = cfg["teacher"], normal.predictor.window
    optimizer = torch.optim.Adam(teacher.parameters(), lr=settings["lr"])
    checkpoint = StageCheckpoint(path, normal, optimizer, key)
    if checkpoint.complete:
        normal.freeze()
        return checkpoint.history
    block_size = min(int(cfg["stage2"]["block_length"]), int(settings.get("block_length", 64)))
    def epoch(record, sequences, training):
        teacher.train(training)
        total = 0.
        blocks_per_batch = (max(length for _, length in sequences) + block_size - 1) // block_size
        progress = Progress(f"teacher {'train' if training else 'validation'} blocks",
            ((len(sequences) + settings["batch_sequences"] - 1) // settings["batch_sequences"]) * blocks_per_batch,
            cfg["runtime"].get("log_interval_seconds", 30))
        completed = 0
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
                completed += 1
                progress.update(completed)
            if training and has_grad:
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1)
                optimizer.step()
                teacher.cell.project()
            total += accumulated * len(batch)
        return total / len(sequences)
    @torch.no_grad()
    def validate():
        """One chronological selection replay, without resetting at compute blocks."""
        teacher.eval()
        direct, closed = None, None
        sums = torch.zeros(normal.schema.channels, dtype=torch.float64, device=normal.schema.amp_mean.device)
        counts = torch.zeros_like(sums)
        total = (len(selection.z) - 1 + block_size - 1) // block_size
        progress = Progress("teacher validation blocks", total, cfg["runtime"].get("log_interval_seconds", 30))
        for start in range(0, len(selection.z) - 1, block_size):
            cooperate()
            length = min(block_size, len(selection.z) - start - 1)
            f = normal.schema.block([selection], [start], length)
            following = normal.schema.block([selection], [start + 1], length)
            hidden, direct = model.teacher_block(f, direct)
            reference, closed = model.teacher_block(f, closed, True)
            now, future = teacher.losses(hidden, f, following)
            rn, rf = teacher.losses(reference, f, following)
            valid = f["sv"] & f["lv"] & following["lv"] & f["target"]
            sums += ((now + future + rn + rf) * .25 * valid).double().sum((0, 1))
            counts += valid.sum((0, 1))
            progress.update(start // block_size + 1)
        if not (counts > 0).any():
            raise ValueError("Normal selection has no usable state-teacher targets.")
        loss = (sums[counts > 0] / counts[counts > 0]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite normal state-teacher validation loss.")
        return float(loss)
    for number in range(len(checkpoint.history), settings["epochs"]):
        if checkpoint.stale >= settings["patience"]:
            break
        began = time.perf_counter()
        sequences = _sequence_starts(fit, w, settings["sequence_length"], settings["train_sequences"], seed + 7200 + number)
        training_loss, validation_loss = epoch(fit, sequences, True), validate()
        checkpoint.record(dict(epoch=number + 1, train_loss=training_loss, validation_loss=validation_loss), validation_loss)
        print(f"teacher epoch={number + 1} train={training_loss:.6f} validation={validation_loss:.6f} "
              f"seconds={time.perf_counter() - began:.1f}", flush=True)
    history = checkpoint.finish()
    normal.freeze()
    return history


@torch.no_grad()
def fit_energy_scales(model, fit, cfg):
    """Complete fitting replay, only block-local descriptors and scalar accumulators."""
    normal = model.normal
    sums = torch.zeros_like(normal.memory.energy_scale, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    state = model.initial_state(1).normal
    length = cfg["stage2"]["block_length"]
    progress = Progress("energy calibration blocks", (len(fit.z) + length - 1) // length,
                        cfg["runtime"].get("log_interval_seconds", 30))
    for start in range(0, len(fit.z), length):
        f = normal.schema.block([fit], [start], min(length, len(fit.z) - start))
        values = model.frozen_block(f, state)
        state = values["final_normal"]
        lavail = values["long_available"]
        sums[:, 1] += (values["long_distance"] * lavail).double().sum((0, 1))
        counts[:, 1] += lavail.sum((0, 1))
        sums[:, 0] += (values["short_distance"] * values["short_available"]).double().sum((0, 1))
        counts[:, 0] += values["short_available"].sum((0, 1))
        cooperate()
        progress.update(start // length + 1)
    normal.memory.energy_scale.copy_((sums / counts.clamp(min=1)).clamp(min=1e-3).float())
    return dict(counts=counts.cpu().tolist(), scales=normal.memory.energy_scale.cpu().tolist())


def build_pairs(model, source, donor, cfg, directory, split, seed, names, relation_tasks=None):
    settings, normal = cfg["stage2"], model.normal
    count = settings["train_pairs"] if split == "train" else settings["validation_pairs"]
    key = fingerprint(dict(split=split, seed=seed, stage2=settings, shape=source.z.shape,
                           normal_key=cfg["_normal_key"], teacher_key=cfg["_teacher_key"]))
    cache = PairCache(Path(directory) / "tasks" / key / split, key, normal.schema.channels,
                         cfg["runtime"].get("task_cache_memory_mb", 256))
    if len(cache) >= count:
        print(f"cache {split} complete pairs={count} reused=True", flush=True)
        return cache
    progress = Progress(f"cache {split} pairs", count, cfg["runtime"].get("log_interval_seconds", 30))
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
            forecast_errors(source, normal, cfg["runtime"]["feature_batch_size"], output=(source_error, source_ready),
                log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 30), progress_label=f"cache {split} source forecast")
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
                pair.clean, clean_error, clean_ready, log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 30),
                progress_label=f"cache {split} pair={index + 1} forecast")
            pair.recipe["forecast_rows_reused"] = reused
            pair.recipe["generator"] = settings["relation_generator"]
            heartbeat = lambda _: progress.update(index)
            pair.clean.routing = model.routing(pair.clean, settings["block_length"], heartbeat)
            pair.corrupt.routing = model.routing(pair.corrupt, settings["block_length"], heartbeat)
            cache.append(pair, (clean_error, corrupt_error), (clean_ready, corrupt_ready))
            progress.update(index + 1, force=index == 0)
    finally:
        source_error._mmap.close()
        source_ready._mmap.close()
    print(f"cache {split} complete pairs={count} storage=packed", flush=True)
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
