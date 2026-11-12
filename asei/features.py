"""Typed, causal descriptors computed per block rather than T*C*38 caches."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import Record
from .tasks import NormalTaskPolicy
from .progress import periodic_progress


class EvidenceLayout:
    """Stable layout: forecast(2), short difference(2), long differences(76)."""
    numeric_stop = 80
    energy_stop = 82
    base_size = 90  # eight availability masks follow the two energies

    def __init__(self, reference_features=False):
        self.size = self.base_size + (2 if reference_features else 0)


class DescriptorSchema(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        self.channels = len(metadata["supports"])
        self.q = 1 + max([len(s) for s in metadata["supports"] if s is not None] + [1])
        self.cat_indices = [i for i, s in enumerate(metadata["supports"]) if s is not None]
        self.num_indices = [i for i in range(self.channels) if i not in self.cat_indices]
        values = torch.zeros(self.channels, self.q)
        known = torch.zeros(self.channels, self.q, dtype=torch.bool)
        for i, support in enumerate(metadata["supports"]):
            if support is not None:
                values[i, 1:len(support) + 1] = torch.tensor(support)
                known[i, 1:len(support) + 1] = True
        allowed = known.clone()
        allowed[:, 0] = True  # Explicit unknown symbol, never nearest-state clipping.
        self.register_buffer("vocabulary", values)
        self.register_buffer("known", known)
        self.register_buffer("allowed", allowed)
        self.register_buffer("categorical", torch.tensor([s is not None for s in metadata["supports"]]))
        self.register_buffer("amp_mean", torch.tensor(metadata["amp_mean"], dtype=torch.float32))
        self.register_buffer("amp_scale", torch.tensor(metadata["amp_scale"], dtype=torch.float32))

    @classmethod
    def fit(cls, record, max_values=16, channel_types=None, channel_names=None, batch_size=1024):
        policy = NormalTaskPolicy(record, max_values, channel_types, channel_names)
        c = record.z.shape[1]
        sums, squares, counts = np.zeros((c, 6)), np.zeros((c, 6)), np.zeros(c)
        for start in range(31, len(record.z), batch_size):
            ix = np.arange(start, min(start + batch_size, len(record.z)))[:, None] + np.arange(-31, 1)
            z, seen = record.z[ix].astype(np.float64), record.mask[ix].all(1)
            mean8, mean32 = z[:, -8:].mean(1), z.mean(1)
            var8, var32 = z[:, -8:].var(1), z.var(1)
            raw = np.stack([z[:, -1], z[:, -1] - z[:, -2], mean8, mean32,
                            np.log(var8 + 1e-4), np.log(var32 + 1e-4)], -1)
            sums += np.where(seen[..., None], raw, 0).sum(0)
            squares += np.where(seen[..., None], raw * raw, 0).sum(0)
            counts += seen.sum(0)
        mean = sums / counts.clip(min=1)[:, None]
        scale = np.sqrt(np.maximum(0, squares / counts.clip(min=1)[:, None] - mean * mean)).clip(min=1e-3)
        scale[counts == 0] = 1
        metadata = dict(supports=[None if s is None else s.tolist() for s in policy.supports],
                        amp_mean=mean.tolist(), amp_scale=scale.tolist(), channels=channel_names)
        return cls(metadata)

    def ids(self, values):
        shape = values.shape
        result = torch.zeros(shape, dtype=torch.long, device=values.device)
        if not self.cat_indices:
            return result
        # Fixed padded vocabularies avoid a CUDA nonzero/synchronization for each
        # categorical channel. Zero remains the explicit unknown symbol.
        matches = (values[..., None] == self.vocabulary) & self.known
        return matches.long().argmax(-1)

    def pack(self, records, starts, length, errors=None, ready=None):
        """CPU-only input preparation, safe to prefetch for independent blocks."""
        values, masks, error_arrays, ready_arrays, target_masks = [], [], [], [], []
        for k, (record, start) in enumerate(zip(records, starts)):
            index = np.arange(start - 31, start + length)
            inside = (index >= 0) & (index < len(record.z))
            clipped = index.clip(0, max(0, len(record.z) - 1))
            values.append(np.where(inside[:, None], record.z[clipped], 0))
            masks.append(record.mask[clipped] & inside[:, None])
            target_index = np.arange(start, start + length)
            target_masks.append((target_index >= record.target_start) & (target_index < len(record.z)))
            clipped_target = target_index.clip(0, len(record.z) - 1)
            error_arrays.append(np.zeros((length, self.channels, 2), np.float32) if errors is None else errors[k][clipped_target])
            ready_arrays.append(np.zeros((length, self.channels, 2), bool) if ready is None else ready[k][clipped_target] & inside[31:, None, None])
        result = dict(z=np.stack(values), mask=np.stack(masks),
                      target=np.stack(target_masks)[..., None],
                      forecast=np.stack(error_arrays), forecast_ready=np.stack(ready_arrays))
        if all(record.routing is not None for record in records):
            for name in ("reference_index", "m_index", "j_index", "short_index", "short_marginal", "peer_cost"):
                rows = []
                for record, start in zip(records, starts):
                    index = np.arange(start, start + length).clip(0, len(record.z) - 1)
                    rows.append(record.routing[name][index])
                result[name] = np.stack(rows)
        return result

    def from_packed(self, packed, start=0, length=None, device=None):
        """GPU descriptor math stays on its owning thread and stream."""
        device = self.amp_mean.device if device is None else device
        length = packed["target"].shape[1] - start if length is None else length
        stop = start + length
        z = torch.as_tensor(packed["z"][:, start:stop + 31], device=device, dtype=torch.float32)
        observed = torch.as_tensor(packed["mask"][:, start:stop + 31], device=device)
        segments = z.unfold(1, 32, 1)  # [B,T,C,32] is a view.
        complete = observed.unfold(1, 32, 1).all(-1)
        mean8, mean32 = segments[..., -8:].mean(-1), segments.mean(-1)
        var8, var32 = segments[..., -8:].var(-1, unbiased=False), segments.var(-1, unbiased=False)
        raw = torch.stack([segments[..., -1], segments[..., -1] - segments[..., -2], mean8, mean32,
                           (var8 + 1e-4).log(), (var32 + 1e-4).log()], -1)
        amp = (raw - self.amp_mean) / self.amp_scale
        shape = (segments - mean32[..., None]) / (var32[..., None] + 1e-4).sqrt()
        chi = torch.cat([amp, shape], -1)
        chi = torch.where(self.categorical[None, None, :, None], torch.zeros_like(chi), chi)
        symbols = self.ids(z).unfold(1, 32, 1)
        short_valid = observed[:, 31:] & observed[:, 30:-1]
        target = torch.as_tensor(packed["target"][:, start:stop], device=device)
        result = dict(chi=chi, v=amp[..., :2].masked_fill(self.categorical[None, None, :, None], 0),
                    tokens=symbols, pair=symbols[..., -2:], sv=short_valid, lv=complete,
                    target=target, observed=observed[:, 31:],
                    forecast=torch.as_tensor(packed["forecast"][:, start:stop], device=device),
                    forecast_ready=torch.as_tensor(packed["forecast_ready"][:, start:stop], device=device))
        if "reference_index" in packed:
            for name in ("reference_index", "m_index", "j_index", "short_index", "short_marginal", "peer_cost"):
                dtype = torch.float32 if name in {"short_marginal", "peer_cost"} else torch.long
                result[name] = torch.as_tensor(packed[name][:, start:stop], device=device, dtype=dtype)
        return result

    def block(self, records, starts, length, errors=None, ready=None, device=None):
        """Build [B,T,C,...] features; negative indices are padding, never wraparound."""
        return self.from_packed(self.pack(records, starts, length, errors, ready), device=device)

    def cluster_features(self, f, channel=None):
        """No ordinal distances for state symbols, including dictionary construction."""
        parts = []
        indices = range(self.channels) if channel is None else [channel]
        for i in indices:
            if i in self.cat_indices:
                sequence = f["tokens"][..., i, :]
                hot = F.one_hot(sequence, self.q).float()
                # Squared Euclidean distance equals 32-step Hamming distance:
                # each unequal symbol contributes two one-hot coordinates / 64.
                parts.append(hot.flatten(-2) / 8.)
            else:
                weight = f["chi"].new_tensor([(.5 / 6) ** .5] * 6 + [(.5 / 32) ** .5] * 32)
                parts.append(f["chi"][..., i, :] * weight)
        return torch.cat(parts, -1)


@torch.no_grad()
def forecast_errors(record, normal, batch_size=64, clean=None, clean_errors=None, clean_ready=None, output=None,
                    log_interval_seconds=0, progress_label="forecast batches", workers=0, prefetch=2):
    """Cache only two errors per channel; affected histories are recomputed exactly."""
    return forecast_errors_many([record], normal, batch_size,
        [clean], [clean_errors], [clean_ready], [output], log_interval_seconds, progress_label, workers, prefetch)[0]


@torch.no_grad()
def forecast_errors_many(records, normal, batch_size=64, cleans=None, clean_errors=None, clean_ready=None, outputs=None,
                         log_interval_seconds=0, progress_label="forecast batches", workers=0, prefetch=2):
    """Fill whole GPU batches across independent tasks; keep exact task-local histories."""
    from .parallel import ordered_prefetch
    count = len(records)
    cleans = [None] * count if cleans is None else cleans
    clean_errors = [None] * count if clean_errors is None else clean_errors
    clean_ready = [None] * count if clean_ready is None else clean_ready
    outputs = [None] * count if outputs is None else outputs
    if not all(len(v) == count for v in (cleans, clean_errors, clean_ready, outputs)):
        raise ValueError("Forecast task lists must have matching lengths.")
    w = normal.predictor.window
    results, tasks = [], []
    for record, clean, cached_error, cached_ready, output in zip(records, cleans, clean_errors, clean_ready, outputs):
        n, c = record.z.shape
        if output is None:
            errors, ready = np.zeros((n, c, 2), np.float32), np.zeros((n, c, 2), bool)
        else:
            errors, ready = output
            if errors.shape != (n, c, 2) or ready.shape != errors.shape:
                raise ValueError("Forecast output cache shape differs from the source.")
            errors[:] = 0
            ready[:] = False
        times, reused = np.arange(w, n), 0
        if clean is not None:
            if clean.z.shape != record.z.shape:
                raise ValueError("Paired prediction reuse requires identical time axes.")
            changed = np.any((clean.mask != record.mask) | (clean.z != record.z), axis=1)
            prefix = np.r_[0, np.cumsum(changed, dtype=np.int64)]
            reuse = (prefix[times + 1] - prefix[times - w]) == 0
            index = times[reuse]
            errors[index], ready[index] = cached_error[index], cached_ready[index]
            times, reused = times[~reuse], len(index)
        results.append((errors, ready, reused))
        tasks.append(times)
    device = normal.schema.amp_mean.device
    normal.predictor.eval()
    offsets = np.arange(-w, 0)
    def batches():
        pending, size = [], 0
        for task, times in enumerate(tasks):
            for left in range(0, len(times), batch_size):
                ix = times[left:left + batch_size]
                while len(ix):
                    take = min(batch_size - size, len(ix))
                    pending.append((task, ix[:take]))
                    ix, size = ix[take:], size + take
                    if size == batch_size:
                        yield pending
                        pending, size = [], 0
        if pending:
            yield pending
    def prepare(batch):
        histories, masks, targets, observed = [], [], [], []
        for task, ix in batch:
            record = records[task]
            history_ix = ix[:, None] + offsets
            histories.append(record.z[history_ix])
            masks.append(record.mask[history_ix])
            targets.append(record.z[ix])
            observed.append(record.mask[ix])
        return batch, np.concatenate(histories), np.concatenate(masks), np.concatenate(targets), np.concatenate(observed)
    with ordered_prefetch(prepare, batches(), workers=min(1, workers), depth=prefetch if workers else 0) as pending:
        for batch, history, masks, target, observed in periodic_progress(pending, progress_label, log_interval_seconds):
            history = torch.as_tensor(history, device=device)
            masks = torch.as_tensor(masks, device=device)
            target = torch.as_tensor(target, device=device)
            prediction, logits, usable = normal.predictor(history, masks)
            residual = target[..., None] - prediction
            symbols = normal.schema.ids(target)
            if normal.schema.cat_indices:
                nll = -F.log_softmax(logits, -1).gather(-1, symbols[..., None, None].expand(-1, -1, 2, 1)).squeeze(-1)
                residual = torch.where(normal.schema.categorical[None, :, None], nll, residual)
            usable &= torch.as_tensor(observed, device=device)[..., None]
            values, available = residual.masked_fill(~usable, 0).cpu().numpy(), usable.cpu().numpy()
            offset = 0
            for task, ix in batch:
                results[task][0][ix] = values[offset:offset + len(ix)]
                results[task][1][ix] = available[offset:offset + len(ix)]
                offset += len(ix)
    return results
