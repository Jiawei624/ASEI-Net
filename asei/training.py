from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from .config import fingerprint, write_json
from .compat import load_torch_checkpoint
from .cache import PackedTaskWriter, TaskFeatureStore
from .data import Record
from .features import extract_features, FeatureSequence, batch_feature_blocks
from .model import NormalRepresentation, DualTrajectoryDetector, segment_arrays
from .progress import periodic_progress
from .tasks import generate_pair, phase_weights
from .resources import cooperate


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class NormalWindows(Dataset):
    def __init__(self, record: Record, window, stride=1):
        self.record, self.window = record, window
        self.indices = np.arange(max(window, record.target_start), len(record.z), stride)
        if not len(self.indices) or not record.mask[self.indices].any():
            raise ValueError("No observed normal targets after the history warmup.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        z, mask = self.record.z, self.record.mask
        return {"history": z[t - self.window:t], "history_mask": mask[t - self.window:t],
                "target": z[t], "target_mask": mask[t], "segment": z[t - 31:t + 1],
                "segment_mask": mask[t - 31:t + 1]}


def complete_windows(record):
    missing = np.concatenate([np.zeros((1, record.mask.shape[1]), np.int64),
                              np.cumsum(~record.mask, axis=0, dtype=np.int64)])
    return missing[32:] == missing[:-32]


@torch.no_grad()
def initialize_prototypes(model: NormalRepresentation, record: Record, log_interval_seconds=0):
    complete = complete_windows(record)
    device = next(model.parameters()).device
    memory = model.memory

    def queries(indices):
        seg, _ = segment_arrays(record, indices)
        chi = model.descriptor(torch.as_tensor(seg, device=device))
        return memory.encode(chi, model.history.identity.weight)

    for c in periodic_progress(range(record.z.shape[1]), "stage1 prototype initialization", log_interval_seconds):
        eligible = np.flatnonzero(complete[:, c]) + 31
        eligible = eligible[eligible >= model.cfg["window"]]
        n = min(len(eligible), memory.km)
        memory.m_active[c].zero_()
        if n:
            indices = eligible[np.linspace(0, len(eligible) - 1, n).astype(int)]
            memory.m_codes[c, :n].copy_(queries(indices)[0][:, c])
            memory.m_active[c, :n] = True
    eligible = np.flatnonzero(complete.all(1)) + 31
    eligible = eligible[eligible >= model.cfg["window"]]
    n = min(len(eligible), memory.kj)
    memory.j_active.zero_()
    if n:
        indices = eligible[np.linspace(0, len(eligible) - 1, n).astype(int)]
        memory.j_codes[:n].copy_(queries(indices)[1])
        memory.j_active[:n] = True


def accumulate_usage(memory, usage, marginal, joint):
    for c in range(memory.channels):
        indices = usage["mi"][:, c][usage["mv"][:, c]]
        marginal[c].scatter_add_(0, indices, torch.ones_like(indices, dtype=marginal.dtype))
    indices = usage["ji"][usage["jv"]]
    joint.scatter_add_(0, indices, torch.ones_like(indices, dtype=joint.dtype))


@torch.no_grad()
def revive_unused(model, optimizer, record, marginal_counts, joint_counts, rng):
    memory = model.memory
    memory.m_unused.copy_(torch.where(marginal_counts > 0, 0, memory.m_unused + 1))
    memory.j_unused.copy_(torch.where(joint_counts > 0, 0, memory.j_unused + 1))
    complete = complete_windows(record)
    device = next(model.parameters()).device

    def encode(indices):
        seg, _ = segment_arrays(record, indices)
        return memory.encode(model.descriptor(torch.as_tensor(seg, device=device)), model.history.identity.weight)

    def clear_momentum(parameter, index):
        for value in optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value[index] = 0

    for c in range(memory.channels):
        dead = torch.where((memory.m_unused[c] >= 5) & memory.m_active[c])[0]
        eligible = np.flatnonzero(complete[:, c]) + 31
        eligible = eligible[eligible >= model.cfg["window"]]
        if len(dead) and len(eligible):
            qm, _ = encode(rng.choice(eligible, len(dead), replace=True))
            memory.m_codes[c, dead] = qm[:, c]
            memory.m_unused[c, dead] = 0
            clear_momentum(memory.m_codes, (c, dead))
    dead = torch.where((memory.j_unused >= 5) & memory.j_active)[0]
    eligible = np.flatnonzero(complete.all(1)) + 31
    eligible = eligible[eligible >= model.cfg["window"]]
    if len(dead) and len(eligible):
        _, qj = encode(rng.choice(eligible, len(dead), replace=True))
        memory.j_codes[dead] = qj
        memory.j_unused[dead] = 0
        clear_momentum(memory.j_codes, dead)


@torch.no_grad()
def finalize_memory(model, record, batch_size=2048, log_interval_seconds=0):
    model.eval()
    memory = model.memory
    marginal, joint = torch.zeros_like(memory.m_unused), torch.zeros_like(memory.j_unused)
    device = next(model.parameters()).device
    for start in periodic_progress(range(model.cfg["window"], len(record.z), batch_size),
                                   "stage1 prototype finalization", log_interval_seconds):
        seg, mask = segment_arrays(record, np.arange(start, min(start + batch_size, len(record.z))))
        chi = model.descriptor(torch.as_tensor(seg, device=device))
        mi, ji = memory.query(chi, torch.as_tensor(mask.all(1), device=device), model.history.identity.weight)
        accumulate_usage(memory, {"mi": mi.clamp_min(0), "ji": ji.clamp_min(0), "mv": mi >= 0, "jv": ji >= 0}, marginal, joint)
    memory.m_active &= marginal > 0
    memory.j_active &= joint > 0
    memory.cache_decoders(model.history.identity.weight)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"marginal_counts": marginal.cpu().tolist(), "joint_counts": joint.cpu().tolist(),
            "active_marginal": memory.m_active.sum(-1).cpu().tolist(),
            "active_joint": int(memory.j_active.sum())}


def fit_representation(model, train, valid, cfg, directory, run_key):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    s = cfg["stage1"]
    interval = cfg["runtime"].get("log_interval_seconds", 300)
    model.descriptor.fit(train)
    initialize_prototypes(model, train, log_interval_seconds=interval)
    # No weight decay for the projected relation or the VQ codebooks.
    special = [model.relation, model.memory.m_codes, model.memory.j_codes]
    special_ids = {id(p) for p in special}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in special_ids], "weight_decay": s["weight_decay"]},
        {"params": special, "weight_decay": 0}], lr=s["lr"])
    pinned = device.type == "cuda"
    workers = cfg["runtime"].get("effective_cpu", {}).get("workers", s["workers"])
    dl = DataLoader(NormalWindows(train, model.cfg["window"], s["stride"]), batch_size=s["batch_size"],
                    shuffle=True, num_workers=workers, drop_last=False, pin_memory=pinned,
                    persistent_workers=workers > 0)
    vl = DataLoader(NormalWindows(valid, model.cfg["window"]), batch_size=s["batch_size"],
                    shuffle=False, num_workers=workers, pin_memory=pinned,
                    persistent_workers=workers > 0)
    rng = np.random.default_rng(8701)
    best, stale, logs = float("inf"), 0, []
    began = time.perf_counter()
    for epoch in range(s["epochs"]):
        epoch_began = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = s["lr"] * min(1, (epoch + 1) / max(1, s["warmup_epochs"]))
        model.train()
        marginal, joint = torch.zeros_like(model.memory.m_unused), torch.zeros_like(model.memory.j_unused)
        train_total, train_n = 0.0, 0
        for batch in periodic_progress(dl, f"stage1 epoch={epoch + 1} train", interval):
            batch = {k: v.to(device, non_blocking=pinned) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, usage = model.normal_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite normal representation loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            model.project_relation()
            accumulate_usage(model.memory, usage, marginal, joint)
            n = len(batch["target"])
            train_total, train_n = train_total + loss.item() * n, train_n + n
        revive_unused(model, optimizer, train, marginal, joint, rng)
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch in periodic_progress(vl, f"stage1 epoch={epoch + 1} validation", interval):
                batch = {k: v.to(device, non_blocking=pinned) for k, v in batch.items()}
                loss, _ = model.normal_loss(batch)
                n = len(batch["target"])
                val_total, val_n = val_total + loss.item() * n, val_n + n
        val_loss = val_total / val_n
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite normal validation loss.")
        logs.append({"epoch": epoch + 1, "train_loss": train_total / train_n, "validation_loss": val_loss})
        print(f"stage1 epoch={epoch + 1} train={train_total / train_n:.6f} validation={val_loss:.6f} "
              f"seconds={time.perf_counter() - epoch_began:.1f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model": model.state_dict(), "model_config": model.cfg, "key": run_key,
                        "epoch": epoch + 1, "validation_loss": best}, directory / "stage1_best.pt")
        else:
            stale += 1
        if stale >= s["patience"]:
            break
    best_checkpoint = load_torch_checkpoint(directory / "stage1_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model"])
    usage = finalize_memory(model, train, log_interval_seconds=interval)
    torch.save({"model": model.state_dict(), "model_config": model.cfg, "key": run_key,
                "epoch": best_checkpoint["epoch"], "validation_loss": best, "usage": usage}, directory / "representation.pt")
    (directory / "stage1_best.pt").unlink()
    elapsed = time.perf_counter() - began
    write_json(directory / "stage1_history.json", {"epochs": logs, "seconds": elapsed, "usage": usage})
    return elapsed


def build_task_cache(model, train, valid, cfg, directory, seed, representation_key):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    s = cfg["stage2"]
    settings = {k: s[k] for k in ("families", "train_pairs", "validation_pairs", "validation_seeds")}
    key = fingerprint({"version": 1, "representation": representation_key, "seed": seed, "tasks": settings})
    manifest_path = directory / "manifest.json"
    previous_seconds = 0.0
    storage = cfg["runtime"].get("task_cache_format", "packed")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["key"] != key:
            raise ValueError("Task cache configuration differs. Use a new output directory.")
        previous_seconds = previous.get("preparation_seconds_total", 0.0)
        # Continue partial legacy caches in their original format.
        storage = previous.get("storage", "npz")
    write_json(manifest_path, {"key": key, "settings": settings, "storage": storage, "complete": False,
                               "preparation_seconds_total": previous_seconds})
    began, recipes, built = time.perf_counter(), [], 0
    for split, source, n in (("train", train, s["train_pairs"]), ("validation", valid, s["validation_pairs"])):
        writer = (PackedTaskWriter(directory / split, key, train.z.shape[1],
                                  cfg["runtime"].get("task_cache_checkpoint_pairs", 16))
                  if storage == "packed" else None)
        try:
            for i in periodic_progress(range(n), f"cache {split} task pairs", cfg["runtime"].get("log_interval_seconds", 300)):
                clean_path, corrupt_path = directory / f"{split}_{i:04d}_clean.npz", directory / f"{split}_{i:04d}_corrupt.npz"
                recipe_path = directory / f"{split}_{i:04d}.json"
                if writer is not None and writer.has_pair(i):
                    recipes.append(writer.recipe(i))
                    continue
                if writer is None and clean_path.exists() and corrupt_path.exists() and recipe_path.exists():
                    recipes.append(json.loads(recipe_path.read_text(encoding="utf-8")))
                    continue
                task_seed = seed + 20000 if split == "train" else s["validation_seeds"][i % len(s["validation_seeds"])]
                pair = generate_pair(source, train, model.cfg["window"], i, task_seed, s["families"])
                built += 1
                clean = extract_features(pair.clean, model, cfg["runtime"]["feature_batch_size"])
                corrupt = extract_features(pair.corrupt, model, cfg["runtime"]["feature_batch_size"])
                clean_valid = clean.mask.any(1) & (np.arange(len(clean)) >= clean.window)
                corrupt_valid = corrupt.mask.any(1) & (np.arange(len(corrupt)) >= corrupt.window)
                cw, pw = phase_weights(pair.labels, clean_valid, corrupt_valid)
                clean.labels, corrupt.labels = np.zeros(len(clean), np.int8), pair.labels
                clean.weights, corrupt.weights = cw, pw
                recipe = {"split": split, **pair.recipe}
                if writer is None:
                    clean.save(clean_path, key, compressed=cfg["runtime"].get("task_cache_compression", False))
                    corrupt.save(corrupt_path, key, compressed=cfg["runtime"].get("task_cache_compression", False))
                    write_json(recipe_path, recipe)
                else:
                    writer.append(i, clean, corrupt, recipe)
                recipes.append(recipe)
        finally:
            if writer is not None:
                writer.close()
        print(f"cache {split} complete pairs={n} storage={storage}", flush=True)
    seconds = time.perf_counter() - began
    size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file() and p.suffix in {".npz", ".bin"})
    write_json(manifest_path, {"key": key, "settings": settings, "storage": storage, "complete": True, "seconds_this_call": seconds,
                               "preparation_seconds_total": previous_seconds + (seconds if built else 0),
                               "bytes": size, "recipes": recipes})
    return key, seconds, size


def _batch_blocks(sequences, start, stop, representation):
    return batch_feature_blocks(sequences, start, stop, representation)


def detector_epoch(detector, representation, cache_dir, cache_key, indices, split, cfg, optimizer=None, epoch=None, store=None):
    owned = store is None
    store = store or TaskFeatureStore(cache_dir, cache_key, cfg["runtime"].get("task_cache_memory_mb", 2048))
    try:
        return _detector_epoch(detector, representation, indices, split, cfg, store, optimizer, epoch)
    finally:
        if owned:
            store.close()


def _detector_epoch(detector, representation, indices, split, cfg, store, optimizer=None, epoch=None):
    training = optimizer is not None
    detector.train(training)
    s = cfg["stage2"]
    device = next(detector.parameters()).device
    total, pairs = 0.0, 0
    label = f"stage2 epoch={epoch} {split}" if epoch is not None else f"stage2 {split}"
    for batch_start in periodic_progress(range(0, len(indices), s["batch_pairs"]), label,
                                         cfg["runtime"].get("log_interval_seconds", 300)):
        batch_indices = indices[batch_start:batch_start + s["batch_pairs"]]
        sequences = [store.get(split, i, role)
                     for i in batch_indices for role in ("clean", "corrupt")]
        state = None
        max_length = max(map(len, sequences))
        for start in range(0, max_length, s["block_length"]):
            cooperate()
            stop = min(start + s["block_length"], max_length)
            block = _batch_blocks(sequences, start, stop, representation)
            labels = np.zeros((len(sequences), stop - start), np.float32)
            weights = np.zeros_like(labels)
            for i, seq in enumerate(sequences):
                length = max(0, min(stop, len(seq)) - start)
                if length:
                    labels[i, :length] = seq.labels[start:start + length]
                    weights[i, :length] = seq.weights[start:start + length]
            has_weights = bool(weights.any())
            labels, weights = torch.as_tensor(labels, device=device), torch.as_tensor(weights, device=device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                if cfg["runtime"].get("detector_execution", "block") == "block":
                    out, state = detector.forward_block(block, state)
                else:
                    outputs = []
                    for t in range(stop - start):
                        out, state = detector.step({k: v[:, t] for k, v in block.items()}, state)
                        outputs.append(out)
                    out = {k: torch.stack([item[k] for item in outputs], 1) for k in outputs[0]}
                if (weights > 0).logical_and(~out["valid"]).any():
                    raise RuntimeError("A task loss weight refers to an unscorable point.")
                loss = (F.binary_cross_entropy_with_logits(out["score"], labels, reduction="none") * weights).sum() / len(batch_indices)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite detector loss.")
                if training and has_weights:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(detector.parameters(), 1)
                    optimizer.step()
            state = state.detach()
            total += loss.item() * len(batch_indices)
        pairs += len(batch_indices)
    return total / pairs


def fit_detector(detector, representation, cfg, cache_dir, cache_key, directory, run_key, seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    s = cfg["stage2"]
    optimizer = torch.optim.Adam(detector.parameters(), lr=s["lr"])
    rng = np.random.default_rng(seed + 10000)
    best, stale, history = float("inf"), 0, []
    began = time.perf_counter()
    store = TaskFeatureStore(cache_dir, cache_key, cfg["runtime"].get("task_cache_memory_mb", 2048))
    try:
        return _fit_detector_epochs(detector, representation, cfg, cache_dir, cache_key, directory,
                                    run_key, rng, optimizer, best, stale, history, began, store)
    finally:
        store.close()


def _fit_detector_epochs(detector, representation, cfg, cache_dir, cache_key, directory,
                          run_key, rng, optimizer, best, stale, history, began, store):
    s = cfg["stage2"]
    for epoch in range(s["epochs"]):
        epoch_began = time.perf_counter()
        train_loss = detector_epoch(detector, representation, cache_dir, cache_key,
                                    rng.permutation(s["train_pairs"]), "train", cfg, optimizer, epoch=epoch + 1, store=store)
        val_loss = detector_epoch(detector, representation, cache_dir, cache_key,
                                  np.arange(s["validation_pairs"]), "validation", cfg, epoch=epoch + 1, store=store)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"stage2 epoch={epoch + 1} train={train_loss:.6f} validation={val_loss:.6f} "
              f"seconds={time.perf_counter() - epoch_began:.1f}", flush=True)
        if val_loss < best:
            best, stale = val_loss, 0
            torch.save({"model": detector.state_dict(), "key": run_key, "epoch": epoch + 1,
                        "validation_loss": best}, directory / "detector.pt")
        else:
            stale += 1
        if stale >= s["patience"]:
            break
    device = next(detector.parameters()).device
    detector.load_state_dict(load_torch_checkpoint(directory / "detector.pt", map_location=device)["model"])
    seconds = time.perf_counter() - began
    write_json(directory / "stage2_history.json", {"epochs": history, "seconds": seconds,
        "complete": True, "key": run_key,
        "parameters": sum(p.numel() for p in detector.parameters()),
        "execution": cfg["runtime"].get("detector_execution", "block"),
        "cache": {"storage": store.storage, "legacy_file_loads": store.loads,
                  "legacy_memory_hits": store.hits, "memory_limit_mb": store.limit / 1024 ** 2}})
    return seconds
