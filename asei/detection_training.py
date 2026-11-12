"""Task-normalized paired TBPTT on a frozen normal model."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .progress import Progress, log, epoch_log
from .resources import cooperate
from .tasks import observable_positive_mask, phase_weights, channel_phase_weights, paired_channel_weights
from .training import StageCheckpoint, _recovery, learning_rate_schedule
from .batching import BatchFeatures
from .parallel import ordered_prefetch, prefetch_depth
from .selection import NormalSelectionReplay, constrained_selection


def paired_ranking_loss(clean, corrupt, weights, valid, margin):
    """Compare the same channel/time/context; both members receive gradients."""
    return (F.softplus(margin + clean - corrupt) * weights * valid).sum() / clean.shape[0]


def prepare_detector_batch(cache, indices, active, settings, schema):
    """Read tasks and build fixed supervision on a single CPU producer."""
    batch = [cache.get(int(i)) for i in indices]
    clean, corrupt, labels, _ = zip(*batch)
    records = [v[0] for v in clean] + [v[0] for v in corrupt]
    errors = [v[1] for v in clean] + [v[1] for v in corrupt]
    ready = [v[2] for v in clean] + [v[2] for v in corrupt]
    size, maximum = len(clean), max(len(r.z) for r in records)
    channels = schema.channels
    whole_weight = np.zeros((2 * size, maximum), np.float32)
    whole_truth = np.zeros_like(whole_weight)
    whole_ramp = np.zeros((size, maximum, channels), np.float32)
    channel_enabled = settings.get("lambda_channel", 0.) > 0 and settings["variant"] != "energy_only"
    channel_weight = np.zeros((2 * size, maximum, channels), np.float32) if channel_enabled else None
    channel_truth = np.zeros_like(channel_weight) if channel_enabled else None
    pair_enabled = settings.get("lambda_pair", 0.) > 0 and settings["variant"] != "energy_only"
    pair_weight = np.zeros((size, maximum, channels), np.float32) if pair_enabled else None
    state_denominators, recovery_denominators = [], []
    for i, (left, right, y) in enumerate(zip(clean, corrupt, labels)):
        def channel_usable(record):
            previous = np.roll(record.mask, 1, axis=0)
            previous[0] = False
            return record.mask & previous & active & (np.arange(len(y)) >= record.target_start)[:, None]
        cv, av = channel_usable(left[0]), channel_usable(right[0])
        positive = observable_positive_mask(left[0], right[0], y)
        try:
            cw, aw = phase_weights(y, cv.any(-1), av.any(-1), positive, settings.get("onset_fraction", .5))
            if channel_enabled:
                channel_positive, ccw, caw = channel_phase_weights(left[0], right[0], y, cv, av, settings.get("onset_fraction", .5))
                channel_weight[i, :len(y)], channel_weight[size + i, :len(y)] = ccw, caw
                channel_truth[size + i, :len(y)] = channel_positive
            if pair_enabled:
                pair_weight[i, :len(y)] = paired_channel_weights(left[0], right[0], y, cv, av, settings.get("onset_fraction", .5))
        except ValueError as exc:
            raise ValueError(f"Invalid cached task index={indices[i]}, recipe={batch[i][3]}: {exc}") from exc
        whole_weight[i, :len(y)], whole_weight[size + i, :len(y)] = cw, aw
        whole_truth[size + i, :len(y)] = y
        ramp = _recovery(left[0], right[0], y)
        whole_ramp[i, :len(y)] = ramp
        state_denominators.append(max(1, cv.sum()))
        recovery_denominators.append(max(1, (ramp * av).sum()))
    supervision = dict(weights=whole_weight, truth=whole_truth, ramps=whole_ramp,
        state_den=np.asarray(state_denominators), recovery_den=np.asarray(recovery_denominators, dtype=np.float32))
    if channel_enabled:
        supervision.update(channel_weights=channel_weight, channel_targets=channel_truth)
    if pair_enabled:
        supervision["pair_weights"] = pair_weight
    packed = schema.pack(records, [0] * len(records), maximum, errors, ready)
    return records, errors, ready, maximum, size, supervision, packed


def fit_detector(model, training, validation, cfg, directory, key, seed, normal_selection=None):
    settings = cfg["stage2"]
    device = model.normal.schema.amp_mean.device
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=settings["lr"], weight_decay=settings.get("weight_decay", 0.))
    rng = np.random.default_rng(seed)
    scheduler = learning_rate_schedule(optimizer, settings)
    checkpoint = StageCheckpoint(Path(directory) / "detector.pt", model, optimizer, key, rng, scheduler)
    if checkpoint.complete:
        return checkpoint.history
    selector = None
    if settings.get("selection_metric", "loss") == "normal_constrained":
        if normal_selection is None:
            raise ValueError("normal_constrained selection requires the normal selection record.")
        selector = NormalSelectionReplay(model, normal_selection, cfg)
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
        totals = torch.zeros(6, device=device, dtype=torch.float64)
        collect_selection = selector is not None and not learn
        selection_rows = []
        def prepare(indices):
            return prepare_detector_batch(cache, indices, active, settings, model.normal.schema)
        with ordered_prefetch(prepare, batches, depth=prefetch_depth(cfg)) as pending:
            for batch_number, batch in enumerate(pending):
                cooperate()
                records, errors, ready, maximum, size, supervision, packed = batch
                batch_features = BatchFeatures(model, records, errors, ready, maximum, cfg, packed=packed)
                tensors = {key: torch.as_tensor(value, device=device) for key, value in supervision.items()}
                weights, truth, ramps = tensors["weights"], tensors["truth"], tensors["ramps"]
                state_den, recovery_den = tensors["state_den"], tensors["recovery_den"]
                channel_weights, channel_targets = tensors.get("channel_weights"), tensors.get("channel_targets")
                channel_enabled = channel_weights is not None
                pair_weights = tensors.get("pair_weights")
                selection_batch = {name: torch.zeros_like(weights, dtype=torch.bool if name.endswith("_valid") else weights.dtype)
                    for name in ("short", "dynamic", "short_valid", "dynamic_valid")} if collect_selection else None
                tail = weights.new_zeros(())
                tail_coefficient = weights.new_zeros(())
                tail_selected = torch.zeros_like(weights, dtype=torch.bool)
                tail_count, tail_active = 1, False
                if tail_enabled and learn:
                    snapshot, snapshot_valid, state = torch.zeros_like(weights), torch.zeros_like(tail_selected), None
                    target_state = None
                    with torch.no_grad():
                        for start in range(0, maximum, block):
                            cooperate()
                            length = min(block, maximum - start)
                            f = batch_features.block(start, length)
                            prepared = model.frozen_block(f, None if state is None else state.normal)
                            out, state = model.forward_block(f, state, prepared=prepared)
                            target_state = batch_features.store(start, f, prepared, target_state, size)
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
                if tail_enabled and not learn:
                    snapshot = torch.zeros_like(weights)
                    snapshot_valid = torch.zeros_like(tail_selected)
                state, teacher_state = None, None
                finite = torch.ones((), dtype=torch.bool, device=device)
                if learn:
                    optimizer.zero_grad(set_to_none=True)
                for start in range(0, maximum, block):
                    cooperate()
                    length = min(block, maximum - start)
                    f = batch_features.block(start, length)
                    cached = batch_features.cached.get(start)
                    with torch.no_grad():
                        if cached is None:
                            targets, teacher_state = model.teacher_block({k: v[:size] for k, v in f.items()}, teacher_state)
                        else:
                            targets, teacher_state = cached["targets"], cached["teacher_state"]
                    with torch.set_grad_enabled(learn):
                        out, state = model.forward_block(f, state, return_aux=True,
                            prepared=None if cached is None else cached["prepared"])
                        if tail_enabled and not learn:
                            snapshot[:, start:start + length] = out["dynamic"]
                            snapshot_valid[:, start:start + length] = out["dynamic_valid"]
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
                        channel_loss = classification.new_zeros(())
                        if channel_enabled:
                            values = F.binary_cross_entropy_with_logits(out["channel_dynamic"], channel_targets[:, start:start + length], reduction="none")
                            channel_loss = (values * channel_weights[:, start:start + length] * out["channel_valid"]).sum() / size
                        pair_loss = classification.new_zeros(())
                        if pair_weights is not None:
                            paired_valid = out["channel_valid"][:size] & out["channel_valid"][size:]
                            pair_loss = paired_ranking_loss(out["channel_dynamic"][:size], out["channel_dynamic"][size:],
                                pair_weights[:, start:start + length], paired_valid, settings["pair_margin"])
                        tail_loss = classification.new_zeros(())
                        if tail_active:
                            positive = (target > 0) & (weight > 0) & valid
                            tail_loss = 2 * (F.softplus(settings["tail_margin"] + tail - out["dynamic"]) * weight * positive).sum() / size
                            # Restore the exact derivative of the task-wide normal tail,
                            # while retaining only this block's trainable graph.
                            selected = tail_selected[:, start:start + length]
                            tail_loss = tail_loss + tail_coefficient * ((out["dynamic"] - out["dynamic"].detach()) * selected).sum() / tail_count
                        loss = classification
                        if channel_enabled:
                            loss = loss + settings["lambda_channel"] * channel_loss
                        if pair_weights is not None:
                            loss = loss + settings["lambda_pair"] * pair_loss
                        if model.variant not in {"no_state", "mlp", "energy_only"}:
                            loss = loss + settings["lambda_state"] * state_loss
                        if model.variant != "no_tail":
                            loss = loss + settings["lambda_tail"] * tail_loss
                        finite &= torch.isfinite(loss.detach())
                        if learn and model.variant != "energy_only":
                            loss.backward()
                        totals += torch.stack([loss, classification, state_loss, tail_loss, channel_loss, pair_loss]).detach().double() * size
                        if collect_selection:
                            for name in selection_batch:
                                selection_batch[name][:, start:start + length] = out[name]
                    state = state.detach()
                    completed += 1
                    progress.update(completed)
                if tail_enabled and not learn:
                    negative = (truth == 0) & (weights > 0) & snapshot_valid
                    positive = (truth > 0) & (weights > 0) & snapshot_valid
                    if bool(negative.any() & positive.any()):
                        values = snapshot[negative]
                        count = max(1, int(np.ceil(settings["tail_fraction"] * len(values))))
                        threshold = values.topk(count).values.mean()
                        penalty = 2 * (F.softplus(settings["tail_margin"] + threshold - snapshot) * weights * positive).sum() / size
                        totals[3] += penalty.double() * size
                        totals[0] += settings["lambda_tail"] * penalty.double() * size
                if not finite:
                    raise FloatingPointError("Non-finite paired detector objective.")
                if collect_selection:
                    selection_rows.append({name: value.cpu().numpy().ravel() for name, value in
                        dict(selection_batch, weight=weights, truth=truth).items()})
                if learn and model.variant != "energy_only":
                    torch.nn.utils.clip_grad_norm_(parameters, 1)
                    optimizer.step()
                    updates += 1
                if batch_number == 0:
                    progress.update(completed, force=True)
        values = (totals / len(cache)).cpu().tolist()
        result = dict(zip(("loss", "classification", "state", "tail", "channel", "pair"), values), optimizer_updates=updates)
        if collect_selection:
            result["_scores"] = {name: np.concatenate([row[name] for row in selection_rows]) for name in selection_rows[0]}
        return result

    for number in range(len(checkpoint.history), settings["epochs"]):
        if checkpoint.stale >= settings["patience"]:
            break
        began = time.perf_counter()
        rate = optimizer.param_groups[0]["lr"]
        train = epoch(training, True)
        train_seconds = time.perf_counter() - began
        validation_began = time.perf_counter()
        valid = epoch(validation, False)
        synthetic_validation_seconds = time.perf_counter() - validation_began
        audit_began = time.perf_counter()
        if selector is not None:
            calibration, audit = selector.evaluate(model)
            valid.update(constrained_selection(valid, valid.pop("_scores"), calibration, audit, settings))
        else:
            valid["objective"] = valid["loss"]
        normal_audit_seconds = time.perf_counter() - audit_began
        if scheduler is not None:
            scheduler.step(valid["objective"])
        checkpoint.record(dict(epoch=number + 1, train=train, selection=valid,
                               train_seconds=train_seconds, synthetic_validation_seconds=synthetic_validation_seconds,
                               normal_audit_seconds=normal_audit_seconds,
                               seconds=time.perf_counter() - began), valid["objective"])
        selection_details = {name: valid[name] for name in ("synthetic_ap", "positive_coverage", "normal_dynamic_fpr", "normal_fused_fpr") if name in valid}
        epoch_log("stage2", number + 1, settings["epochs"], train["loss"], valid["loss"], began, cfg, checkpoint,
                  lr=rate, classification=train["classification"],
                  state=train["state"], tail=train["tail"], channel=train["channel"], pair=train["pair"],
                  train_seconds=train_seconds, synthetic_validation_seconds=synthetic_validation_seconds,
                  normal_audit_seconds=normal_audit_seconds,
                  selection_objective=valid["objective"], **selection_details)
    history = checkpoint.finish()
    model.eval()
    return history
