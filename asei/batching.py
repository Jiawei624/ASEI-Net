"""Bounded batch-local feature reuse; no cross-task or cross-model cache."""
from __future__ import annotations

import torch
import numpy as np

from .parallel import ordered_prefetch, prefetch_depth


def tensor_bytes(values):
    return sum(value.numel() * value.element_size() for value in values.values() if torch.is_tensor(value))


class BatchFeatures:
    def __init__(self, model, records, errors, ready, maximum, cfg, starts=None, packed=None):
        self.model, self.records, self.errors, self.ready = model, records, errors, ready
        self.starts = [0] * len(records) if starts is None else list(starts)
        self.packed = packed
        device = model.normal.schema.amp_mean.device
        requested = float(cfg["runtime"].get("batch_working_memory_mb", 2048)) * 1024 ** 2
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            self.limit = min(requested, free * .25)
        else:
            self.limit = min(requested, 64 * 1024 ** 2)
        self.full, self.cached, self.used, self.closed = None, {}, 0, False
        # Conservative allowance includes unfolded views and temporary descriptors.
        estimate = len(records) * maximum * model.normal.schema.channels * 640
        if estimate <= self.limit * .5:
            self.full = (model.normal.schema.block(records, self.starts, maximum, errors, ready) if packed is None
                         else model.normal.schema.from_packed(packed))
            self.used = tensor_bytes(self.full)

    def block(self, start, length):
        if self.full is not None:
            return {key: value[:, start:start + length] for key, value in self.full.items()}
        if start in self.cached:
            return self.cached[start]["features"]
        if self.packed is not None:
            return self.model.normal.schema.from_packed(self.packed, start, length)
        return self.model.normal.schema.block(self.records, [value + start for value in self.starts], length, self.errors, self.ready)

    def store(self, start, features, prepared, teacher_state, clean_size):
        if self.closed:
            return teacher_state
        size = tensor_bytes(prepared) + (0 if self.full is not None else tensor_bytes(features))
        size += 2 * clean_size * features["v"].shape[1] * self.model.normal.schema.channels * 32 * 4
        if self.used + size > self.limit:
            self.closed = True
            return teacher_state
        targets, final = self.model.teacher_block({key: value[:clean_size] for key, value in features.items()}, teacher_state)
        self.cached[start] = dict(features=features, prepared=prepared, targets=targets, teacher_state=final)
        self.used += size
        return final


def teacher_scoring_mask(packed, window):
    """The same full-task scoring mask, computed once without a second GPU pass."""
    mask = packed["mask"]
    missing = np.pad(np.cumsum(~mask, axis=1, dtype=np.int32), ((0, 0), (1, 0), (0, 0)))
    complete = (missing[:, 32:] - missing[:, :-32]) == 0
    short = mask[:, 31:] & mask[:, 30:-1]
    target = packed["target"][:, :-1] & (np.arange(complete.shape[1] - 1)[None, :, None] >= window)
    return short[:, :-1] & complete[:, :-1] & complete[:, 1:] & target


def teacher_denominator(packed, window):
    return teacher_scoring_mask(packed, window).sum(1)


def feature_blocks(schema, records, starts, maximum, block, cfg, errors=None, ready=None):
    """Prefetch CPU inputs only; consume dependent recurrent states in order."""
    def prepare(offset):
        length = min(block, maximum - offset)
        packed = schema.pack(records, [start + offset for start in starts], length, errors, ready)
        return offset, length, packed
    with ordered_prefetch(prepare, range(0, maximum, block), depth=prefetch_depth(cfg)) as pending:
        for offset, length, packed in pending:
            yield offset, length, schema.from_packed(packed)
