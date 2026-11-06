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
from .features import extract_features, FeatureSequence, batch_feature_blocks, slice_normal_features, feature_summary
from .model import NormalRepresentation, DualTrajectoryDetector, ReferenceConsistentDetector, PrototypeMemory, segment_arrays
from .progress import periodic_progress
from .tasks import (generate_pair, phase_weights, observable_positive_mask,
                    TASK_SUPERVISION_PROTOCOL, NormalTaskPolicy, generate_reference_pair,
                    observable_channel_mask, REFERENCE_TASK_PROTOCOL)
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
    if hasattr(model, "energy_scale"):
        sums = torch.zeros_like(model.energy_scale, dtype=torch.float64)
        counts = torch.zeros_like(sums)
        sample_indices = np.unique(np.linspace(model.cfg["window"], len(record.z) - 1,
                                               min(2048, len(record.z) - model.cfg["window"])).astype(int))
        sampled_energy, sampled_valid = [], []
        for start in periodic_progress(range(model.cfg["window"], len(record.z), batch_size),
                                       "stage1 energy normalization", log_interval_seconds):
            seg, mask = segment_arrays(record, np.arange(start, min(start + batch_size, len(record.z))))
            chi = model.descriptor(torch.as_tensor(seg, device=device))
            mi, ji = memory.query(chi, torch.as_tensor(mask.all(1), device=device), model.history.identity.weight)
            rm, rj, mv, jv = memory.references(mi, ji)
            valid = torch.stack([mv, jv], -1)
            energy = torch.stack([memory.reconstruction_loss(rm, chi), memory.reconstruction_loss(rj, chi)], -1)
            sums += (energy.double() * valid).sum(0)
            counts += valid.sum(0)
            selected = sample_indices[(sample_indices >= start) & (sample_indices < start + len(energy))] - start
            if len(selected):
                selected = torch.as_tensor(selected, device=device)
                sampled_energy.append(energy[selected].cpu().numpy())
                sampled_valid.append(valid[selected].cpu().numpy())
        model.energy_scale.copy_((sums / counts.clamp_min(1)).clamp_min(1e-3).float())
        normalized = np.concatenate(sampled_energy) / model.energy_scale.cpu().numpy()
        usable = np.concatenate(sampled_valid)
        quantiles = [.5, .9, .99, 1.]
        normal_energy = {"sample_points": len(normalized), "quantiles": quantiles,
            "sources": ["marginal", "joint"], "valid_sample_counts": usable.sum(0).tolist(),
            "normalized_quantiles_by_channel_source": [[np.quantile(normalized[:, c, s][usable[:, c, s]], quantiles).tolist()
                if usable[:, c, s].any() else [None] * len(quantiles) for s in range(2)]
                for c in range(normalized.shape[1])]}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"marginal_counts": marginal.cpu().tolist(), "joint_counts": joint.cpu().tolist(),
            "active_marginal": memory.m_active.sum(-1).cpu().tolist(),
            "active_joint": int(memory.j_active.sum()),
            **({"normal_energy": normal_energy} if hasattr(model, "energy_scale") else {})}


def normal_objective(sums, counts):
    values = (sums / counts.clamp_min(1)).detach().cpu().tolist()
    return sum(values), dict(zip(("forecast", "marginal", "joint"), values))


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
        train_sums = torch.zeros(3, dtype=torch.float64, device=device)
        train_counts = torch.zeros_like(train_sums)
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
            train_sums += usage["normal_loss_sums"].double()
            train_counts += usage["normal_loss_counts"]
        revive_unused(model, optimizer, train, marginal, joint, rng)
        model.eval()
        val_sums, val_counts = torch.zeros_like(train_sums), torch.zeros_like(train_counts)
        with torch.no_grad():
            for batch in periodic_progress(vl, f"stage1 epoch={epoch + 1} validation", interval):
                batch = {k: v.to(device, non_blocking=pinned) for k, v in batch.items()}
                loss, usage = model.normal_loss(batch)
                val_sums += usage["normal_loss_sums"].double()
                val_counts += usage["normal_loss_counts"]
        val_loss, val_components = normal_objective(val_sums, val_counts)
        train_loss, train_components = normal_objective(train_sums, train_counts)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite normal validation loss.")
        logs.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
                     "train_components": train_components, "validation_components": val_components,
                     "validation_counts": val_counts.cpu().tolist()})
        print(f"stage1 epoch={epoch + 1} train={train_loss:.6f} validation={val_loss:.6f} "
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


def task_cache_key(representation_key, seed, settings):
    return fingerprint({"version": 2, "supervision": TASK_SUPERVISION_PROTOCOL,
                        "representation": representation_key, "seed": seed, "tasks": settings})


def task_settings(cfg):
    s = cfg["stage2"]
    settings = {k: s[k] for k in ("families", "train_pairs", "validation_pairs", "validation_seeds")}
    if s.get("method") == "in_1_2_3":
        settings.update(task_protocol=REFERENCE_TASK_PROTOCOL,
            policy={k: s[k] for k in ("hard_negative_fraction", "discrete_max_values", "donor_candidates", "boundary_blend")},
            channel_types=cfg["data"].get("channel_types", {}))
    return settings


def build_task_cache(model, train, valid, cfg, directory, seed, representation_key, channel_names=None, clean_features=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    s = cfg["stage2"]
    settings = task_settings(cfg)
    policy = (NormalTaskPolicy(train, s["discrete_max_values"], cfg["data"].get("channel_types"), channel_names)
              if s.get("method") == "in_1_2_3" else None)
    key = task_cache_key(representation_key, seed, settings)
    manifest_path = directory / "manifest.json"
    previous_seconds = 0.0
    previous = {}
    storage = cfg["runtime"].get("task_cache_format", "packed")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["key"] != key:
            raise ValueError("Task cache configuration differs. Use a new output directory.")
        previous_seconds = previous.get("preparation_seconds_total", 0.0)
        # Continue partial legacy caches in their original format.
        storage = previous.get("storage", "npz")
    write_json(manifest_path, {"key": key, "settings": settings, "storage": storage, "complete": False,
                               "supervision_protocol": TASK_SUPERVISION_PROTOCOL,
                               "preparation_seconds_total": previous_seconds})
    began, recipes, built = time.perf_counter(), [], 0
    availability = previous.get("normal_reference_availability", {})
    for split, source, n in (("train", train, s["train_pairs"]), ("validation", valid, s["validation_pairs"])):
        shared_normal = clean_features.get(split) if clean_features else None
        reuse = policy is not None and cfg["runtime"].get("reuse_clean_features", True)
        hard_anchors = policy.hard_anchors(source, model.cfg["window"]) if policy is not None else None
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
                if policy is None:
                    pair = generate_pair(source, train, model.cfg["window"], i, task_seed, s["families"])
                else:
                    pair = generate_reference_pair(source, train, model.cfg["window"], i, task_seed, s["families"],
                        policy, hard_anchors, s["hard_negative_fraction"], s["donor_candidates"], s["boundary_blend"])
                built += 1
                feature_statistics = {}
                if reuse:
                    if shared_normal is None:
                        shared_normal = extract_features(source, model, cfg["runtime"]["feature_batch_size"],
                            log_interval_seconds=cfg["runtime"].get("log_interval_seconds", 300),
                            progress_label=split + " shared normal features")
                    if split not in availability:
                        availability[split] = feature_summary(shared_normal)
                    clean = slice_normal_features(shared_normal, pair.recipe["source_start"], pair.recipe["source_stop"])
                    if not np.array_equal(clean.z, pair.clean.z) or not np.array_equal(clean.mask, pair.clean.mask):
                        raise RuntimeError("Normal feature reuse refers to different inputs or masks.")
                    corrupt = extract_features(pair.corrupt, model, cfg["runtime"]["feature_batch_size"],
                                               reference=clean, statistics=feature_statistics)
                    feature_statistics["shared_clean_targets"] = max(0, len(clean) - clean.window)
                else:
                    clean = extract_features(pair.clean, model, cfg["runtime"]["feature_batch_size"])
                    corrupt = extract_features(pair.corrupt, model, cfg["runtime"]["feature_batch_size"])
                clean_valid = clean.mask.any(1) & (np.arange(len(clean)) >= clean.window)
                corrupt_valid = corrupt.mask.any(1) & (np.arange(len(corrupt)) >= corrupt.window)
                positive_valid = observable_positive_mask(pair.clean, pair.corrupt, pair.labels)
                cw, pw = phase_weights(pair.labels, clean_valid, corrupt_valid, positive_valid)
                clean.labels, corrupt.labels = np.zeros(len(clean), np.int8), pair.labels
                clean.weights, corrupt.weights = cw, pw
                recipe = {"split": split, **pair.recipe, "source_channels": train.z.shape[1],
                          "feature_reuse": feature_statistics,
                          "supervision_protocol": TASK_SUPERVISION_PROTOCOL,
                          "unidentifiable_positive_points": int(np.count_nonzero(
                              pair.labels.astype(bool) & corrupt_valid & ~positive_valid))}
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
                               "supervision_protocol": TASK_SUPERVISION_PROTOCOL,
                               "preparation_seconds_total": previous_seconds + (seconds if built else 0),
                               "bytes": size, "recipes": recipes, "task_summary": summarize_tasks(recipes),
                               "normal_reference_availability": availability})
    return key, seconds, size


def summarize_tasks(recipes):
    result = {}
    for split in sorted({recipe["split"] for recipe in recipes}):
        selected = [recipe for recipe in recipes if recipe["split"] == split]
        lengths = [event["stop"] - event["start"] for recipe in selected for event in recipe["events"]]
        costs = [event["donor_match_cost"] for recipe in selected for event in recipe["events"] if "donor_match_cost" in event]
        retries = [recipe.get("attempt", 0) + recipe.get("support_retry", 0) for recipe in selected]
        channel_counts = [len(event["channels"]) for recipe in selected for event in recipe["events"] if "channels" in event]
        fractions = [len(event["channels"]) / recipe["source_channels"] for recipe in selected if "source_channels" in recipe
                     for event in recipe["events"] if "channels" in event]
        result[split] = dict(pairs=len(selected), family_counts={family: sum(r.get("family") == family for r in selected)
            for family in sorted({r.get("family", "unknown") for r in selected})},
            event_length_quantiles=np.quantile(lengths, [0, .25, .5, .75, 1]).tolist() if lengths else [],
            retry_fraction=float(np.mean(np.asarray(retries) > 0)), maximum_retry=max(retries, default=0),
            hard_source_fraction=float(np.mean([r.get("hard_normal_source", False) for r in selected])),
            mean_donor_match_cost=float(np.mean(costs)) if costs else None,
            affected_channel_count_quantiles=np.quantile(channel_counts, [0, .5, 1]).tolist() if channel_counts else [],
            mean_affected_channel_fraction=float(np.mean(fractions)) if fractions else None,
            feature_reuse={name: sum(r.get("feature_reuse", {}).get(name, 0) for r in selected)
                for name in ("predicted_targets", "queried_targets", "reused_predictions", "reused_queries", "shared_clean_targets")})
    return result


def _batch_blocks(sequences, start, stop, representation):
    return batch_feature_blocks(sequences, start, stop, representation)


def relation_region_masks(recipe, observable):
    """Task validation regions; only observable changes count as positives."""
    regions = {key: np.zeros(len(observable), bool) for key in ("boundary", "interior")}
    if recipe.get("family") == "relation":
        for event in recipe["events"]:
            start, stop = event["start"], event["stop"]
            left, right = event.get("interior", [start, stop])
            regions["interior"][left:right] = True
            regions["boundary"][start:left] = True
            regions["boundary"][right:stop] = True
    return {name: mask & observable for name, mask in regions.items()}


def paired_region_metrics(samples):
    from sklearn.metrics import average_precision_score
    result = {"scope": "Fixed synthetic selection tasks; matched clean/corrupt points, balanced classes."}
    for name, values in samples.items():
        scores = np.concatenate(values, axis=1) if values else np.empty((2, 0))
        labels = np.repeat([0, 1], scores.shape[1])
        result[name] = {"paired_points": scores.shape[1], "auc_pr": float(average_precision_score(labels, scores.ravel()))
                        if scores.shape[1] else None}
    return result


def detector_epoch(detector, representation, cache_dir, cache_key, indices, split, cfg, optimizer=None, epoch=None, store=None, teacher=None):
    owned = store is None
    store = store or TaskFeatureStore(cache_dir, cache_key, cfg["runtime"].get("task_cache_memory_mb", 2048))
    try:
        return _detector_epoch(detector, representation, indices, split, cfg, store, optimizer, epoch, teacher)
    finally:
        if owned:
            store.close()


def _detector_epoch(detector, representation, indices, split, cfg, store, optimizer=None, epoch=None, teacher=None):
    training = optimizer is not None
    detector.train(training)
    s = cfg["stage2"]
    device = next(detector.parameters()).device
    total, pairs = 0.0, 0
    modern = isinstance(detector, ReferenceConsistentDetector)
    components = {name: 0. for name in ("bce", "reference", "separation", "tail")}
    updates = 0
    tail_counts = dict(negative_candidates=0, selected=0, blocks=0)
    region_samples = {key: [] for key in ("boundary", "interior")}
    task_batch_update = s.get("optimizer_update", "block") == "task_batch"
    label = f"stage2 epoch={epoch} {split}" if epoch is not None else f"stage2 {split}"
    for batch_start in periodic_progress(range(0, len(indices), s["batch_pairs"]), label,
                                         cfg["runtime"].get("log_interval_seconds", 300)):
        batch_indices = indices[batch_start:batch_start + s["batch_pairs"]]
        sequences = [store.get(split, i, role)
                     for i in batch_indices for role in ("clean", "corrupt")]
        state = None
        teacher_state = None
        batch_has_gradients = False
        if training and task_batch_update:
            optimizer.zero_grad(set_to_none=True)
        max_length = max(map(len, sequences))
        if modern:
            affected = []
            regions = []
            reference_counts, separation_counts = [], []
            for j, index in enumerate(batch_indices):
                clean, corrupt = sequences[2 * j:2 * j + 2]
                recipe = store.recipe(split, index)
                mask = observable_channel_mask(clean, corrupt, corrupt.labels, recipe["events"])
                if not training:
                    regions.append(relation_region_masks(recipe, mask.any(-1)))
                reference_valid = np.ones_like(clean.mask)
                for sequence in (clean, corrupt):
                    available = np.zeros_like(sequence.mask)
                    if "marginal" in detector.sources:
                        available |= sequence.marginal_index >= 0
                    if "joint" in detector.sources:
                        available |= sequence.joint_index[:, None] >= 0
                    reference_valid &= available & sequence.mask & (np.arange(len(sequence)) >= sequence.window)[:, None]
                reference_counts.append(max(1, reference_valid.sum()))
                separation_counts.append(max(1, (mask & reference_valid).sum()))
                affected.append(mask)
            reference_counts = torch.as_tensor(reference_counts, device=device)
            separation_counts = torch.as_tensor(separation_counts, device=device)
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
            if training and not task_batch_update:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                if cfg["runtime"].get("detector_execution", "block") == "block":
                    out, state = detector.forward_block(block, state, **({"return_aux": True} if modern else {}))
                else:
                    outputs = []
                    for t in range(stop - start):
                        out, state = detector.step({k: v[:, t] for k, v in block.items()}, state,
                                                   **({"return_aux": True} if modern else {}))
                        outputs.append(out)
                    out = {k: torch.stack([item[k] for item in outputs], 1) for k in outputs[0]}
                if (weights > 0).logical_and(~out["valid"]).any():
                    raise RuntimeError("A task loss weight refers to an unscorable point.")
                if modern and not training:
                    for j, masks in enumerate(regions):
                        for name, region in masks.items():
                            points = np.flatnonzero(region[start:stop])
                            if len(points):
                                points = torch.as_tensor(points, device=device)
                                usable = out["valid"][2 * j, points] & out["valid"][2 * j + 1, points]
                                region_samples[name].append(out["score"][2 * j:2 * j + 2, points[usable]].cpu().numpy())
                bce = (F.binary_cross_entropy_with_logits(out["score"], labels, reduction="none") * weights).sum() / len(batch_indices)
                auxiliary = {name: bce.new_zeros(()) for name in ("reference", "separation", "tail")}
                if modern:
                    valid_reference = out["reference_valid"][::2] & out["reference_valid"][1::2]
                    if teacher is not None:
                        with torch.no_grad():
                            target, teacher_state, teacher_valid = teacher.forward_block({k: v[::2] for k, v in block.items()}, teacher_state)
                        valid_reference &= teacher_valid
                        quality = (1 / (1 + out["energy"][::2])).detach()
                        normal = out["normal_state"]
                        ref_loss = F.huber_loss(normal[::2], target, reduction="none").mean(-1)
                        ref_loss += F.huber_loss(normal[1::2], target, reduction="none").mean(-1)
                        auxiliary["reference"] = ((ref_loss * quality * valid_reference).sum((1, 2)) / reference_counts).mean()
                    mask = np.zeros((len(batch_indices), stop - start, block["chi"].shape[-2]), bool)
                    for j, full_mask in enumerate(affected):
                        length = max(0, min(stop, len(full_mask)) - start)
                        if length:
                            mask[j, :length] = full_mask[start:start + length]
                    mask = torch.as_tensor(mask, device=device) & valid_reference
                    delta = out["state_delta"].square().mean(-1)
                    distance = PrototypeMemory.reconstruction_loss(block["chi"][::2], block["chi"][1::2])
                    margin = s["gamma_max"] * distance / (1 + distance)
                    auxiliary["separation"] = ((F.relu(margin + delta[::2] - delta[1::2]) * mask).sum((1, 2)) / separation_counts).mean()
                    if detector.variant in {"mlp", "single_gru"}:
                        auxiliary["separation"] = bce.new_zeros(())
                    negatives = out["score"][(labels == 0) & (weights > 0) & out["valid"]]
                    if negatives.numel():
                        k = max(1, int(np.ceil(s["tail_fraction"] * negatives.numel())))
                        tail_counts["negative_candidates"] += negatives.numel()
                        tail_counts["selected"] += k
                        tail_counts["blocks"] += 1
                        tail = negatives.topk(k).values.mean()
                        auxiliary["tail"] = (F.softplus(s["tail_margin"] + tail - out["score"]) * weights * labels).sum() * 2 / len(batch_indices)
                loss = bce
                for name, setting, disabled in (("reference", "lambda_ref", "no_teacher"),
                        ("separation", "lambda_sep", "no_separation"), ("tail", "lambda_tail", "no_tail")):
                    if modern and detector.variant != disabled:
                        loss = loss + s[setting] * auxiliary[name]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite detector loss.")
                if training and has_weights:
                    loss.backward()
                    batch_has_gradients = True
                    if not task_batch_update:
                        torch.nn.utils.clip_grad_norm_(detector.parameters(), 1)
                        optimizer.step()
                        updates += 1
                        if modern:
                            detector.project()
            state = state.detach()
            if teacher_state is not None:
                teacher_state = teacher_state.detach()
            total += loss.item() * len(batch_indices)
            for name, value in dict(bce=bce, **auxiliary).items():
                components[name] += value.item() * len(batch_indices)
        if training and task_batch_update and batch_has_gradients:
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1)
            optimizer.step()
            updates += 1
            if modern:
                detector.project()
        pairs += len(batch_indices)
    detector.epoch_components = {name: value / pairs for name, value in components.items()}
    detector.epoch_statistics = {"optimizer_updates": updates, "optimizer_update": s.get("optimizer_update", "block"),
                                "local_tail": tail_counts}
    detector.validation_regions = paired_region_metrics(region_samples) if modern and not training else {}
    return total / pairs


def fit_detector(detector, representation, cfg, cache_dir, cache_key, directory, run_key, seed, teacher=None):
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
                                    run_key, rng, optimizer, best, stale, history, began, store, teacher)
    finally:
        store.close()


def _fit_detector_epochs(detector, representation, cfg, cache_dir, cache_key, directory,
                          run_key, rng, optimizer, best, stale, history, began, store, teacher=None):
    s = cfg["stage2"]
    for epoch in range(s["epochs"]):
        epoch_began = time.perf_counter()
        train_loss = detector_epoch(detector, representation, cache_dir, cache_key,
                                    rng.permutation(s["train_pairs"]), "train", cfg, optimizer, epoch=epoch + 1, store=store, teacher=teacher)
        train_components = dict(detector.epoch_components)
        train_statistics = dict(detector.epoch_statistics)
        val_loss = detector_epoch(detector, representation, cache_dir, cache_key,
                                  np.arange(s["validation_pairs"]), "validation", cfg, epoch=epoch + 1, store=store, teacher=teacher)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss,
                        "train_components": train_components, "validation_components": dict(detector.epoch_components)})
        history[-1]["train_statistics"] = train_statistics
        if detector.validation_regions:
            history[-1]["validation_regions"] = detector.validation_regions
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
