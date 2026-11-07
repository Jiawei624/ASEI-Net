"""Task-normalized paired TBPTT on a frozen normal model."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .progress import Progress
from .resources import cooperate
from .tasks import observable_positive_mask, phase_weights
from .training import StageCheckpoint, _recovery


def fit_detector(model, training, validation, cfg, directory, key, seed):
    settings = cfg["stage2"]
    device = model.normal.schema.amp_mean.device
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=settings["lr"])
    rng = np.random.default_rng(seed)
    checkpoint = StageCheckpoint(Path(directory) / "detector.pt", model, optimizer, key, rng)
    if checkpoint.complete:
        return checkpoint.history
    active = model.normal.memory.m_active.any(-1).cpu().numpy()
    block = settings["block_length"]

    def epoch(cache, learn):
        model.train(learn)
        # Length buckets reduce padding without dropping or resampling any task.
        order = rng.permutation(len(cache)) if learn else np.arange(len(cache))
        order = sorted(order, key=lambda i: cache.index["entries"][i]["length"])
        batches = [order[i:i + settings["batch_pairs"]] for i in range(0, len(order), settings["batch_pairs"])]
        if learn:
            rng.shuffle(batches)
        total_blocks = sum((max(cache.index["entries"][i]["length"] for i in batch) + block - 1) // block for batch in batches)
        tail_enabled = model.variant not in {"no_tail", "energy_only"} and settings["lambda_tail"] > 0
        progress = Progress(f"stage2 {'train' if learn else 'validation'} blocks", total_blocks * (2 if tail_enabled else 1),
                            cfg["runtime"].get("log_interval_seconds", 30))
        completed, updates = 0, 0
        totals = torch.zeros(4, device=device, dtype=torch.float64)
        for batch_number, indices in enumerate(batches):
            cooperate()
            batch = [cache.get(int(i)) for i in indices]
            clean, corrupt, labels, _ = zip(*batch)
            records = [v[0] for v in clean] + [v[0] for v in corrupt]
            errors = [v[1] for v in clean] + [v[1] for v in corrupt]
            ready = [v[2] for v in clean] + [v[2] for v in corrupt]
            size = len(clean)
            maximum = max(len(r.z) for r in records)
            whole_weight = np.zeros((2 * size, maximum), np.float32)
            whole_truth = np.zeros_like(whole_weight)
            whole_ramp = np.zeros((size, maximum, model.normal.schema.channels), np.float32)
            state_denominators, recovery_denominators = [], []
            for i, (left, right, y) in enumerate(zip(clean, corrupt, labels)):
                def channel_usable(record):
                    previous = np.roll(record.mask, 1, axis=0)
                    previous[0] = False
                    return record.mask & previous & active & (np.arange(len(y)) >= record.target_start)[:, None]
                cv, av = channel_usable(left[0]), channel_usable(right[0])
                positive = observable_positive_mask(left[0], right[0], y)
                cw, aw = phase_weights(y, cv.any(-1), av.any(-1), positive)
                whole_weight[i, :len(y)], whole_weight[size + i, :len(y)] = cw, aw
                whole_truth[size + i, :len(y)] = y
                ramp = _recovery(left[0], right[0], y)
                whole_ramp[i, :len(y)] = ramp
                state_denominators.append(max(1, cv.sum()))
                recovery_denominators.append(max(1, (ramp * av).sum()))
            weights = torch.as_tensor(whole_weight, device=device)
            truth = torch.as_tensor(whole_truth, device=device)
            ramps = torch.as_tensor(whole_ramp, device=device)
            state_den = torch.as_tensor(state_denominators, device=device)
            recovery_den = torch.as_tensor(recovery_denominators, device=device)
            tail = weights.new_zeros(())
            tail_coefficient = weights.new_zeros(())
            tail_selected = torch.zeros_like(weights, dtype=torch.bool)
            tail_count, tail_active = 1, False
            if tail_enabled:
                snapshot, snapshot_valid, state = torch.zeros_like(weights), torch.zeros_like(tail_selected), None
                with torch.no_grad():
                    for start in range(0, maximum, block):
                        cooperate()
                        length = min(block, maximum - start)
                        f = model.normal.schema.block(records, [start] * len(records), length, errors, ready)
                        out, state = model.forward_block(f, state)
                        snapshot[:, start:start + length] = out["dynamic"]
                        snapshot_valid[:, start:start + length] = out["dynamic_valid"]
                        completed += 1
                        progress.update(completed)
                    negative = (truth == 0) & (weights > 0) & snapshot_valid
                    positive = (truth > 0) & (weights > 0) & snapshot_valid
                    tail_active = bool(negative.any() & positive.any())
                    if tail_active:
                        positions = negative.flatten().nonzero(as_tuple=False).flatten()
                        tail_count = max(1, int(np.ceil(settings["tail_fraction"] * len(positions))))
                        selected = positions[snapshot.flatten()[positions].topk(tail_count).indices]
                        tail_selected.flatten()[selected] = True
                        tail = snapshot.flatten()[selected].mean()
                        tail_coefficient = 2 * (torch.sigmoid(settings["tail_margin"] + tail - snapshot)
                                                * weights * positive).sum() / size
            state, teacher_state = None, None
            if learn:
                optimizer.zero_grad(set_to_none=True)
            for start in range(0, maximum, block):
                cooperate()
                length = min(block, maximum - start)
                f = model.normal.schema.block(records, [start] * len(records), length, errors, ready)
                with torch.no_grad():
                    targets, teacher_state = model.teacher_block({k: v[:size] for k, v in f.items()}, teacher_state)
                with torch.set_grad_enabled(learn):
                    out, state = model.forward_block(f, state, return_aux=True)
                    weight = weights[:, start:start + length]
                    target = truth[:, start:start + length]
                    ramp = ramps[:, start:start + length]
                    valid = out["dynamic_valid"]
                    classification = (F.binary_cross_entropy_with_logits(out["dynamic"], target, reduction="none") * weight * valid).sum() / size
                    eligible = out["reference_valid"] & f["target"]
                    clean_loss = F.huber_loss(out["observed"][:size], targets, reduction="none").mean(-1)
                    recovery_loss = F.huber_loss(out["observed"][size:], targets, reduction="none").mean(-1)
                    anchored = (clean_loss * eligible[:size]).sum((1, 2)) / state_den
                    recovered = (recovery_loss * ramp * eligible[size:]).sum((1, 2)) / recovery_den
                    state_loss = (anchored + recovered).mean()
                    tail_loss = classification.new_zeros(())
                    if tail_active:
                        positive = (target > 0) & (weight > 0) & valid
                        tail_loss = 2 * (F.softplus(settings["tail_margin"] + tail - out["dynamic"]) * weight * positive).sum() / size
                        # Restore the exact derivative of the task-wide normal tail,
                        # while retaining only this block's trainable graph.
                        selected = tail_selected[:, start:start + length]
                        tail_loss = tail_loss + tail_coefficient * ((out["dynamic"] - out["dynamic"].detach()) * selected).sum() / tail_count
                    loss = classification
                    if model.variant not in {"no_state", "mlp", "energy_only"}:
                        loss = loss + settings["lambda_state"] * state_loss
                    if model.variant != "no_tail":
                        loss = loss + settings["lambda_tail"] * tail_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite paired detector objective.")
                    if learn and model.variant != "energy_only":
                        loss.backward()
                    totals += torch.stack([loss, classification, state_loss, tail_loss]).detach().double() * size
                state = state.detach()
                completed += 1
                progress.update(completed)
            if learn and model.variant != "energy_only":
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                updates += 1
            if batch_number == 0:
                progress.update(completed, force=True)
        values = (totals / len(cache)).cpu().tolist()
        return dict(zip(("loss", "classification", "state", "tail"), values), optimizer_updates=updates)

    for number in range(len(checkpoint.history), settings["epochs"]):
        if checkpoint.stale >= settings["patience"]:
            break
        began = time.perf_counter()
        print(f"stage2 epoch={number + 1} start train_pairs={len(training)} validation_pairs={len(validation)}", flush=True)
        train, valid = epoch(training, True), epoch(validation, False)
        checkpoint.record(dict(epoch=number + 1, train=train, selection=valid), valid["loss"])
        print(f"stage2 epoch={number + 1} train={train['loss']:.6f} validation={valid['loss']:.6f} "
              f"classification={train['classification']:.6f} state={train['state']:.6f} "
              f"tail={train['tail']:.6f} seconds={time.perf_counter() - began:.1f}", flush=True)
    history = checkpoint.finish()
    model.eval()
    return history
