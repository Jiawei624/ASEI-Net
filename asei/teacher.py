"""Normal-only teacher pretraining on causal sequence batches."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from .compat import load_torch_checkpoint
from .config import write_json
from .features import FeatureSequence, batch_feature_blocks
from .progress import periodic_progress
from .state import NormalStateTeacher, balanced_description_loss


def make_teacher(representation, stage2):
    return NormalStateTeacher(representation.history.identity.weight,
        **{key: stage2[key] for key in ("cell", "eta", "rho", "sources") if key in stage2})


def _sequences(features, count, length, seed):
    w = features.window
    length = min(int(length), len(features) - w)
    if length < 2:
        raise ValueError("Teacher data needs at least two normal targets after history warmup.")
    rng = np.random.default_rng(seed)
    starts = rng.integers(w, len(features) - length + 1, size=count)
    fields = ("z", "mask", "prediction", "marginal_index", "joint_index")
    return [FeatureSequence(**{key: getattr(features, key)[start - w:start + length] for key in fields}, window=w)
            for start in starts]


def _usable(sequence, teacher):
    available = np.zeros_like(sequence.mask)
    if "marginal" in teacher.sources:
        available |= sequence.marginal_index >= 0
    if "joint" in teacher.sources:
        available |= sequence.joint_index[:, None] >= 0
    return available & sequence.mask & (np.arange(len(sequence)) >= sequence.window)[:, None]


def teacher_epoch(teacher, representation, sequences, cfg, optimizer=None):
    training = optimizer is not None
    teacher.train(training)
    batch_size, block_length = cfg["teacher"]["batch_sequences"], cfg["stage2"]["block_length"]
    total = 0.
    task_batch_update = cfg["teacher"].get("optimizer_update", "block") == "task_batch"
    updates = 0
    for left in periodic_progress(range(0, len(sequences), batch_size), "normal teacher batches",
                                 cfg["runtime"].get("log_interval_seconds", 300)):
        batch = sequences[left:left + batch_size]
        valid = [_usable(seq, teacher) for seq in batch]
        device = next(teacher.parameters()).device
        current_counts = torch.tensor([max(1, mask.sum()) for mask in valid], device=device)
        next_counts = torch.tensor([max(1, (mask[:-1] & mask[1:]).sum()) for mask in valid], device=device)
        state = None
        batch_has_gradients = False
        if training and task_batch_update:
            optimizer.zero_grad(set_to_none=True)
        for start in range(0, max(map(len, batch)), block_length):
            stop = min(start + block_length, max(map(len, batch)))
            f = batch_feature_blocks(batch, start, stop, representation)
            next_f = batch_feature_blocks(batch, start + 1, stop + 1, representation)
            if training and not task_batch_update:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                hidden, state, usable = teacher.forward_block(f, state)
                next_available = next_f["observed"] & next_f["prediction_ready"]
                next_available &= ((next_f["m_valid"] & ("marginal" in teacher.sources)) |
                                   (next_f["j_valid"] & ("joint" in teacher.sources)))
                current_loss = balanced_description_loss(teacher.current_head(hidden), f["chi"])
                next_loss = balanced_description_loss(teacher.next_head(hidden), next_f["chi"])
                loss = .5 * ((current_loss * usable).sum((1, 2)) / current_counts +
                             (next_loss * usable * next_available).sum((1, 2)) / next_counts).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite normal teacher loss.")
                if training and usable.any():
                    loss.backward()
                    batch_has_gradients = True
                    if not task_batch_update:
                        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1)
                        optimizer.step()
                        teacher.project()
                        updates += 1
            state = state.detach()
            total += loss.item() * len(batch)
        if training and task_batch_update and batch_has_gradients:
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1)
            optimizer.step()
            teacher.project()
            updates += 1
    teacher.epoch_statistics = {"optimizer_updates": updates, "optimizer_update": cfg["teacher"].get("optimizer_update", "block")}
    return total / len(sequences)


def fit_teacher(teacher, representation, train_features, selection_features, cfg, directory, key, seed):
    from .training import seed_everything
    seed_everything(seed + 7000)
    settings = cfg["teacher"]
    directory = Path(directory)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=settings["lr"])
    validation = _sequences(selection_features, settings["validation_sequences"], settings["sequence_length"], 7101)
    if not any(_usable(seq, teacher).any() for seq in validation):
        raise ValueError("Teacher selection data has no usable normal reference.")
    best, stale, logs = float("inf"), 0, []
    began = time.perf_counter()
    for epoch in range(settings["epochs"]):
        train = _sequences(train_features, settings["train_sequences"], settings["sequence_length"], seed + 7200 + epoch)
        if not any(_usable(seq, teacher).any() for seq in train):
            raise ValueError("Teacher training data has no usable normal reference.")
        train_loss = teacher_epoch(teacher, representation, train, cfg, optimizer)
        train_statistics = dict(teacher.epoch_statistics)
        validation_loss = teacher_epoch(teacher, representation, validation, cfg)
        logs.append(dict(epoch=epoch + 1, train_loss=train_loss, validation_loss=validation_loss, train_statistics=train_statistics))
        print(f"teacher epoch={epoch + 1} train={train_loss:.6f} validation={validation_loss:.6f}", flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save(dict(model=teacher.state_dict(), key=key, epoch=epoch + 1, validation_loss=best), directory / "teacher.pt")
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    teacher.load_state_dict(load_torch_checkpoint(directory / "teacher.pt", map_location=next(teacher.parameters()).device)["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    write_json(directory / "teacher_history.json", dict(key=key, complete=True, epochs=logs, seconds=time.perf_counter() - began))
