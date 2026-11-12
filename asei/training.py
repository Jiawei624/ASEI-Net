"""Normal training, fixed evidence preparation and resumable checkpoints."""
from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .compat import load_torch_checkpoint
from .config import fingerprint, write_json
from .tasks import NormalTaskPolicy, generate_reference_pair, TaskGenerationError, validate_pair_observability
from .resources import cooperate
from .progress import Progress, log, epoch_log
from .features import forecast_errors, forecast_errors_many
from .cache import PairCache
from .batching import BatchFeatures, teacher_scoring_mask, feature_blocks
from .parallel import ordered_prefetch, preparation_workers, prefetch_depth, background_writer
from .tasks import normal_difficulty, typed_donor_matcher, scale_numeric_intervention


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


def learning_rate_schedule(optimizer, settings):
    if not settings.get("lr_patience", 0):
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
        factor=settings.get("lr_factor", .5), patience=settings["lr_patience"],
        min_lr=settings.get("min_lr", 1e-5))


class StageCheckpoint:
    """Best weights plus resumable last-epoch state; an early best is not completion."""
    def __init__(self, path, module, optimizer, key, rng=None, scheduler=None):
        self.path, self.module, self.optimizer = Path(path), module, optimizer
        self.last = self.path.with_name(self.path.stem + "_last.pt")
        self.key, self.rng = key, rng
        self.scheduler = scheduler
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
                if scheduler is not None and "lr_scheduler" in saved:
                    scheduler.load_state_dict(saved["lr_scheduler"])
                torch.set_rng_state(saved["torch_rng"].cpu())
                if torch.cuda.is_available() and "cuda_rng" in saved:
                    torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
                if rng is not None:
                    rng.bit_generator.state = saved["numpy_rng"]
                log(f"resume {self.path.stem} epoch={len(self.history) + 1}")

    def record(self, entry, validation):
        self.history.append(entry)
        improved = validation < self.best
        self.best, self.stale = (validation, 0) if improved else (self.best, self.stale + 1)
        saved = dict(key=self.key, model=self.module.state_dict(), optimizer=self.optimizer.state_dict(),
                     history=self.history, best=self.best, stale=self.stale, complete=False,
                     torch_rng=torch.get_rng_state())
        if self.rng is not None:
            saved["numpy_rng"] = self.rng.bit_generator.state
        if self.scheduler is not None:
            saved["lr_scheduler"] = self.scheduler.state_dict()
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
        for name in ("optimizer", "torch_rng", "cuda_rng", "numpy_rng", "lr_scheduler"):
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
    scheduler = learning_rate_schedule(optimizer, settings)
    checkpoint = StageCheckpoint(path, normal, optimizer, key, rng, scheduler)
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
        rate = optimizer.param_groups[0]["lr"]
        training_loss, validation_loss = epoch(fit, True), epoch(selection, False)
        if scheduler is not None:
            scheduler.step(validation_loss)
        checkpoint.record(dict(epoch=number + 1, train_loss=training_loss, validation_loss=validation_loss,
                               seconds=time.perf_counter() - began), validation_loss)
        epoch_log("stage1", number + 1, settings["epochs"], training_loss, validation_loss, began, cfg, checkpoint,
                  lr=rate)
    history = checkpoint.history if checkpoint.complete else checkpoint.finish()
    normal.freeze()
    # Dictionaries contain real fitting windows and are fixed before teacher learning.
    saved = load_torch_checkpoint(path, map_location=normal.schema.amp_mean.device)
    if not saved.get("memory_complete", False):
        normal.memory.fit(fit, seed, cfg["memory"]["pool_size"], cfg["memory"]["rare_fraction"], interval,
                          workers=preparation_workers(cfg))
        normal.memory.set_peers(predictor.weights())
        saved.update(model=normal.state_dict(), memory_complete=True)
        _save(path, saved)
        log("memory construction complete", cfg)
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
    optimizer = torch.optim.Adam(teacher.parameters(), lr=settings["lr"], weight_decay=settings.get("weight_decay", 0.))
    scheduler = learning_rate_schedule(optimizer, settings)
    checkpoint = StageCheckpoint(path, normal, optimizer, key, scheduler=scheduler)
    if checkpoint.complete:
        normal.freeze()
        return checkpoint.history
    block_size = min(int(cfg["stage2"]["block_length"]), int(settings.get("block_length", 64)))
    def epoch(record, sequences, training):
        teacher.train(training)
        total = torch.zeros((), dtype=torch.float64, device=normal.schema.amp_mean.device)
        blocks_per_batch = (max(length for _, length in sequences) + block_size - 1) // block_size
        progress = Progress(f"teacher {'train' if training else 'validation'} blocks",
            ((len(sequences) + settings["batch_sequences"] - 1) // settings["batch_sequences"]) * blocks_per_batch,
            cfg["runtime"].get("log_interval_seconds", 30))
        completed = 0
        def prepare(first):
            batch = sequences[first:first + settings["batch_sequences"]]
            starts, lengths = zip(*batch)
            packed = normal.schema.pack([record] * len(batch), starts, max(lengths) + 1)
            return batch, starts, lengths, packed, teacher_scoring_mask(packed, w)
        with ordered_prefetch(prepare, range(0, len(sequences), settings["batch_sequences"]),
                              depth=prefetch_depth(cfg)) as pending:
            for batch, starts, lengths, packed, scoring in pending:
                features = BatchFeatures(model, [record] * len(batch), None, None, max(lengths) + 1, cfg, starts=starts, packed=packed)
                def current_and_next(offset, length):
                    full = features.block(offset, length + 1)
                    f = {key: value[:, :-1] for key, value in full.items()}
                    following = {key: value[:, 1:] for key, value in full.items()}
                    if offset == 0:
                        f["sv"][:, 0] = False
                    f["target"] &= (torch.arange(offset, offset + length, device=f["v"].device)[None, :, None] >= w)
                    return f, following
                direct, closed = None, None
                if training:
                    optimizer.zero_grad(set_to_none=True)
                accumulated, has_grad = total.new_zeros(()), False
                # Fixed per-sequence/channel denominators avoid block-size dependent targets.
                denominator = torch.as_tensor(scoring.sum(1), device=normal.schema.amp_mean.device, dtype=torch.float32)
                for offset in range(0, max(lengths), block_size):
                    cooperate()
                    length = min(block_size, max(lengths) - offset)
                    f, following = current_and_next(offset, length)
                    valid = f["sv"] & f["lv"] & following["lv"] & f["target"]
                    with torch.set_grad_enabled(training):
                        hidden, direct = model.teacher_block(f, direct, False)
                        reference_hidden, closed = model.teacher_block(f, closed, True)
                        now, future = teacher.losses(hidden, f, following)
                        rn, rf = teacher.losses(reference_hidden, f, following)
                        channel_loss = ((now + future + rn + rf) * .25 * valid).sum(1) / denominator.clamp(min=1)
                        loss = (channel_loss.sum(-1) / (denominator > 0).sum(-1).clamp(min=1)).mean()
                        if training and scoring[:, offset:offset + length].any():
                            loss.backward()
                            has_grad = True
                    direct, closed = direct.detach(), closed.detach()
                    accumulated += loss.detach().double()
                    completed += 1
                    progress.update(completed)
                # One synchronization per task batch. No optimizer update occurs
                # before all of its blocks have passed this finite-loss check.
                if not torch.isfinite(accumulated):
                    raise FloatingPointError("Non-finite normal state-teacher loss.")
                if training and has_grad:
                    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1)
                    optimizer.step()
                    teacher.cell.project()
                total += accumulated * len(batch)
        return float(total / len(sequences))
    @torch.no_grad()
    def validate():
        """One chronological selection replay, without resetting at compute blocks."""
        teacher.eval()
        direct, closed = None, None
        sums = torch.zeros(normal.schema.channels, dtype=torch.float64, device=normal.schema.amp_mean.device)
        counts = torch.zeros_like(sums)
        total = (len(selection.z) - 1 + block_size - 1) // block_size
        progress = Progress("teacher validation blocks", total, cfg["runtime"].get("log_interval_seconds", 30))
        # Include the following descriptor in each CPU input block; overlapping
        # descriptors do not reset either recurrent trajectory.
        def prepare(start):
            length = min(block_size, len(selection.z) - start - 1)
            return start, length, normal.schema.pack([selection], [start], length + 1)
        with ordered_prefetch(prepare, range(0, len(selection.z) - 1, block_size), depth=prefetch_depth(cfg)) as pending:
            for start, length, packed in pending:
                cooperate()
                full = normal.schema.from_packed(packed)
                f = {key: value[:, :-1] for key, value in full.items()}
                following = {key: value[:, 1:] for key, value in full.items()}
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
        rate = optimizer.param_groups[0]["lr"]
        sequences = _sequence_starts(fit, w, settings["sequence_length"], settings["train_sequences"], seed + 7200 + number)
        training_loss = epoch(fit, sequences, True)
        train_seconds = time.perf_counter() - began
        validation_began = time.perf_counter()
        validation_loss = validate()
        validation_seconds = time.perf_counter() - validation_began
        if scheduler is not None:
            scheduler.step(validation_loss)
        checkpoint.record(dict(epoch=number + 1, train_loss=training_loss, validation_loss=validation_loss,
                               train_seconds=train_seconds, validation_seconds=validation_seconds,
                               seconds=time.perf_counter() - began), validation_loss)
        epoch_log("teacher", number + 1, settings["epochs"], training_loss, validation_loss, began, cfg, checkpoint,
                  lr=rate, train_seconds=train_seconds, validation_seconds=validation_seconds)
    history = checkpoint.finish()
    normal.freeze()
    return history


@torch.no_grad()
def fit_energy_scales(model, fit, cfg):
    """Complete fitting replay, only block-local descriptors and scalar accumulators."""
    normal = model.normal
    sums = torch.zeros_like(normal.memory.energy_scale, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    support_sum = torch.zeros(normal.schema.channels, dtype=torch.float64, device=sums.device)
    support_square, support_count = torch.zeros_like(support_sum), torch.zeros_like(support_sum)
    normal.memory.reference_limit.fill_(float("inf"))
    state = model.initial_state(1).normal
    length = cfg["stage2"]["block_length"]
    progress = Progress("energy calibration blocks", (len(fit.z) + length - 1) // length,
                        cfg["runtime"].get("log_interval_seconds", 30))
    for start, _, f in feature_blocks(normal.schema, [fit], [0], len(fit.z), length, cfg):
        values = model.frozen_block(f, state)
        state = values["final_normal"]
        lavail = values["long_available"]
        sums[:, 1] += (values["long_distance"] * lavail).double().sum((0, 1))
        counts[:, 1] += lavail.sum((0, 1))
        sums[:, 0] += (values["short_distance"] * values["short_available"]).double().sum((0, 1))
        counts[:, 0] += values["short_available"].sum((0, 1))
        support_valid = values["external"] & f["sv"]
        support = values["peer_cost"].double()
        support_sum += (support * support_valid).sum((0, 1))
        support_square += (support.square() * support_valid).sum((0, 1))
        support_count += support_valid.sum((0, 1))
        cooperate()
        progress.update(start // length + 1)
    normal.memory.energy_scale.copy_((sums / counts.clamp(min=1)).clamp(min=1e-3).float())
    multiplier = cfg["stage2"].get("reference_guard_multiplier", 0.)
    if multiplier:
        mean = support_sum / support_count.clamp(min=1)
        std = (support_square / support_count.clamp(min=1) - mean.square()).clamp(min=0).sqrt()
        limit = (mean + multiplier * std).clamp(min=1e-3)
        normal.memory.reference_limit.copy_(torch.where(support_count > 0, limit, torch.full_like(limit, float("inf"))).float())
    memory = normal.memory
    memory.local_energy_scale.copy_(memory.energy_scale[..., None].expand_as(memory.local_energy_scale))
    memory.local_energy_count.zero_()
    local_statistics = dict(enabled=bool(cfg["stage2"].get("local_scaling", False)))
    if local_statistics["enabled"]:
        # A second normal-only replay uses the fitted support guard. Each radius
        # describes the exact retrieval rule used at inference, including fallback.
        local_sum = torch.zeros_like(memory.local_energy_scale, dtype=torch.float64)
        prior = cfg["stage2"]["local_scale_prior"]
        clip = cfg["stage2"]["local_scale_clip"]
        global_scale = memory.energy_scale.double()[..., None]
        state = model.initial_state(1).normal
        for _, _, f in feature_blocks(normal.schema, [fit], [0], len(fit.z), length, cfg):
            values = model.frozen_block(f, state)
            state = values["final_normal"]
            distance = torch.stack([values["short_distance"], values["long_distance"]], -2)
            available = torch.stack([values["short_available"], values["long_available"]], -2)
            # [B,T,C,span,source] -> [C,span,source,B*T]. Symbolic
            # channels accumulate mismatch rates, never distances between IDs.
            index = values["energy_indices"].permute(2, 3, 4, 0, 1).reshape(*memory.energy_scale.shape, -1)
            valid = available.permute(2, 3, 4, 0, 1).reshape_as(index)
            distance = distance.permute(2, 3, 4, 0, 1).reshape_as(index).double()
            clipped = torch.minimum(distance, clip * global_scale).masked_fill(~valid, 0)
            local_sum.scatter_add_(-1, index, clipped)
            memory.local_energy_count.scatter_add_(-1, index, valid.long())
        radius = (local_sum + prior * global_scale) / (memory.local_energy_count + prior)
        ratio = (radius / global_scale).clamp(cfg["stage2"]["local_scale_min"], cfg["stage2"]["local_scale_max"])
        memory.local_energy_scale.copy_((ratio * global_scale).float())
        occupied = memory.local_energy_count > 0
        used = ratio[occupied]
        local_statistics.update(method="clipped_mean_with_global_pseudocount", prior_count=prior,
            clip_ratio=clip, min_ratio=cfg["stage2"]["local_scale_min"], max_ratio=cfg["stage2"]["local_scale_max"],
            occupied_groups=int(occupied.sum()), samples=int(memory.local_energy_count.sum()),
            scale_ratio_quantiles=torch.quantile(used, used.new_tensor([0., .1, .5, .9, 1.])).cpu().tolist() if len(used) else [],
            occupied_groups_by_channel=occupied.sum((1, 2, 3)).cpu().tolist())
    return dict(counts=counts.cpu().tolist(), scales=normal.memory.energy_scale.cpu().tolist(),
                local_scaling=local_statistics,
                reference_support_counts=support_count.cpu().tolist(),
                reference_limits=[float(v) if torch.isfinite(v) else None for v in normal.memory.reference_limit.cpu()])


def build_pairs(model, source, donor, cfg, directory, split, seed, names, relation_tasks=None):
    settings, normal = cfg["stage2"], model.normal
    count = settings["train_pairs"] if split == "train" else settings["validation_pairs"]
    key = fingerprint(dict(split=split, seed=seed, stage2=settings, shape=source.z.shape,
                           normal_key=cfg["_normal_key"], teacher_key=cfg["_teacher_key"]))
    cache = PairCache(Path(directory) / "tasks" / key / split, key, normal.schema.channels,
                         cfg["runtime"].get("task_cache_memory_mb", 256))
    if len(cache) >= count:
        log(f"cache {split} complete pairs={count} reused=True", cfg)
        return cache
    began = time.perf_counter()
    timings = dict(source_forecast_seconds=0., generation_wait_seconds=0., task_forecast_seconds=0., routing_seconds=0., write_seconds=0.)
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
        source_began = time.perf_counter()
        shape = (*source.z.shape, 2)
        source_error = np.lib.format.open_memmap(error_path, mode="w+", dtype=np.float32, shape=shape)
        source_ready = np.lib.format.open_memmap(ready_path, mode="w+", dtype=bool, shape=shape)
        try:
            forecast_errors(source, normal, cfg["runtime"]["feature_batch_size"], output=(source_error, source_ready),
                log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 30), progress_label=f"cache {split} source forecast",
                workers=preparation_workers(cfg), prefetch=prefetch_depth(cfg))
            source_error.flush()
            source_ready.flush()
        finally:
            source_error._mmap.close()
            source_ready._mmap.close()
        write_json(marker, dict(key=key, shape=shape))
        timings["source_forecast_seconds"] = time.perf_counter() - source_began
    source_error, source_ready = np.load(error_path, mmap_mode="r"), np.load(ready_path, mmap_mode="r")
    active = normal.memory.m_active.any(-1).cpu().numpy()
    cycle = 2 * len([f for f in settings["families"] if source.z.shape[1] > 1 or f != "relation"])

    def generate(index):
        # All random draws are local to the task identity. Worker scheduling
        # cannot change another task, its family/layout, or the cache order.
        for retry in range(64):
            generation_index = index + retry * cycle
            pair = generate_reference_pair(source, donor, normal.predictor.window, generation_index, seed,
                settings["families"], policy, hard, settings["hard_negative_fraction"],
                settings["donor_candidates"], settings["boundary_blend"], relation_matcher=matcher)
            pair.recipe["index"] = index
            if retry:
                pair.recipe["final_observability_retry"] = retry
                pair.recipe["generation_index"] = generation_index
            if relation_tasks is not None:
                pair = relation_tasks.apply(pair, seed)
            strengths = settings.get("task_strengths", [1.])
            pair = scale_numeric_intervention(pair, policy.supports, strengths[(index // cycle) % len(strengths)])
            try:
                validate_pair_observability(pair, active)
            except TaskGenerationError:
                continue
            return pair
        raise TaskGenerationError(f"Cannot prepare a scorable task after 64 final checks: split={split}, index={index}, seed={seed}.")

    try:
        group_size = max(1, int(cfg["runtime"].get("preparation_batch_pairs", 8)))
        workers = preparation_workers(cfg)
        def persist(group):
            started = time.perf_counter()
            for pair, error, valid in group:
                cache.append(pair, error, valid)
            timings["write_seconds"] += time.perf_counter() - started
        with background_writer(persist, cfg["runtime"].get("async_task_writes", False)) as write_group, ordered_prefetch(
                generate, range(len(cache), count), workers=workers, depth=2 * group_size if workers else 0) as generated:
            while True:
                started = time.perf_counter()
                pairs = list(islice(generated, group_size))
                timings["generation_wait_seconds"] += time.perf_counter() - started
                if not pairs:
                    break
                clean_errors, clean_masks = [], []
                for pair in pairs:
                    begin, end = pair.recipe["source_start"], pair.recipe["source_stop"]
                    clean_error, clean_ready = source_error[begin:end].copy(), source_ready[begin:end].copy()
                    clean_error[:normal.predictor.window] = 0
                    clean_ready[:normal.predictor.window] = False
                    clean_errors.append(clean_error)
                    clean_masks.append(clean_ready)
                started = time.perf_counter()
                forecasts = forecast_errors_many([pair.corrupt for pair in pairs], normal, cfg["runtime"]["feature_batch_size"],
                    [pair.clean for pair in pairs], clean_errors, clean_masks, workers=workers, prefetch=prefetch_depth(cfg))
                timings["task_forecast_seconds"] += time.perf_counter() - started
                pending = []
                for pair, clean_error, clean_ready, (corrupt_error, corrupt_ready, reused) in zip(pairs, clean_errors, clean_masks, forecasts):
                    pair.recipe["forecast_rows_reused"] = reused
                    pair.recipe["generator"] = settings["relation_generator"]
                    pending.append((pair, (clean_error, corrupt_error), (clean_ready, corrupt_ready)))
                records = [p[0].clean for p in pending] + [p[0].corrupt for p in pending]
                started = time.perf_counter()
                routes = model.routing_many(records, settings["block_length"])
                timings["routing_seconds"] += time.perf_counter() - started
                for record, routing in zip(records, routes):
                    record.routing = routing
                write_group(pending)
    finally:
        source_error._mmap.close()
        source_ready._mmap.close()
    cache.preparation_timings = timings
    log(f"cache {split} complete pairs={count} storage=packed seconds={time.perf_counter() - began:.1f} " +
        " ".join(f"{k}={v:.2f}" for k, v in timings.items()), cfg)
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
