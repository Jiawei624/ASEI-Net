"""Sequential task-feature storage and bounded reuse of legacy cache files."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from .features import FeatureSequence


_DTYPES = {"z": np.float32, "mask": np.bool_, "prediction": np.float32,
           "marginal_index": np.int32, "joint_index": np.int32,
           "labels": np.int8, "weights": np.float32}


def _shape(name, channels):
    if name == "prediction":
        return (channels, 2)
    return (channels,) if name in {"z", "mask", "marginal_index"} else ()


def _atomic_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class PackedTaskWriter:
    """Append pairs sequentially; commit an index only after both records are written."""

    def __init__(self, directory, key, channels, checkpoint_pairs=16):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "index.json"
        self.checkpoint_pairs = max(1, int(checkpoint_pairs))
        self.record = {"format": 1, "key": key, "channels": channels, "items": [], "recipes": []}
        if self.path.exists():
            self.record = json.loads(self.path.read_text(encoding="utf-8"))
            if self.record.get("format") != 1 or self.record.get("key") != key or self.record.get("channels") != channels:
                raise ValueError("Packed task cache does not match the frozen representation.")
        self.total = sum(item["length"] for item in self.record["items"])
        self.handles = {}
        try:
            for name, dtype in _DTYPES.items():
                path = self.directory / (name + ".bin")
                expected = self.total * int(np.prod(_shape(name, channels))) * np.dtype(dtype).itemsize
                actual = path.stat().st_size if path.exists() else 0
                if actual < expected:
                    raise ValueError("A committed packed-cache field is truncated: " + str(path))
                handle = open(path, "r+b" if path.exists() else "w+b", buffering=0)
                self.handles[name] = handle
                # A process interrupted before index commit may leave an uncommitted tail.
                handle.truncate(expected)
                handle.seek(expected)
        except Exception:
            self.close(commit=False)
            raise

    def has_pair(self, index):
        return index < len(self.record["recipes"])

    def recipe(self, index):
        return self.record["recipes"][index]

    def append(self, index, clean, corrupt, recipe):
        if index != len(self.record["recipes"]):
            raise ValueError("Packed task pairs must be appended in order.")
        items, offset = [], self.total
        for role, sequence in (("clean", clean), ("corrupt", corrupt)):
            for name, dtype in _DTYPES.items():
                value = getattr(sequence, name)
                expected = (len(sequence), *_shape(name, self.record["channels"]))
                if value is None or value.shape != expected:
                    raise ValueError("Invalid task-cache field: " + name)
                np.ascontiguousarray(value, dtype=dtype).tofile(self.handles[name])
            items.append({"offset": offset, "length": len(sequence), "window": sequence.window,
                          "index": index, "role": role})
            offset += len(sequence)
        for handle in self.handles.values():
            handle.flush()
        record = dict(self.record, items=self.record["items"] + items,
                      recipes=self.record["recipes"] + [recipe])
        self.record, self.total = record, offset
        if len(record["recipes"]) % self.checkpoint_pairs == 0:
            _atomic_json(self.path, record)

    def close(self, commit=True):
        if commit and self.handles:
            _atomic_json(self.path, self.record)
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(commit=exc[0] is None)


class PackedTaskReader:
    def __init__(self, directory, key):
        directory = Path(directory)
        self.record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if self.record.get("format") != 1 or self.record.get("key") != key:
            raise ValueError("Stale packed task cache.")
        total = sum(item["length"] for item in self.record["items"])
        if not total:
            raise ValueError("Packed task cache is empty.")
        self.fields = {}
        for name, dtype in _DTYPES.items():
            shape = (total, *_shape(name, self.record["channels"]))
            path = directory / (name + ".bin")
            if path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
                raise ValueError("Packed task-cache field has an unexpected size: " + str(path))
            self.fields[name] = np.memmap(path, mode="r", dtype=dtype, shape=shape)

    def get(self, index, role):
        item = self.record["items"][2 * index + (1 if role == "corrupt" else 0)]
        if item["index"] != index or item["role"] != role:
            raise ValueError("Packed task-cache index is inconsistent.")
        start, stop = item["offset"], item["offset"] + item["length"]
        fields = {name: array[start:stop] for name, array in self.fields.items()}
        return FeatureSequence(window=item["window"], **fields)

    def close(self):
        # Outstanding sequence views keep their mappings alive until released.
        self.fields.clear()


class TaskFeatureStore:
    """Memory-map packed arrays, or cache legacy NPZ records within a RAM limit."""

    def __init__(self, directory, key, memory_mb=2048):
        self.directory, self.key = Path(directory), key
        manifest = self.directory / "manifest.json"
        record = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        self.storage = record.get("storage", "npz")
        self.limit = int(float(memory_mb) * 1024 ** 2)
        self.cache, self.readers = OrderedDict(), {}
        self.bytes, self.hits, self.loads = 0, 0, 0

    def get(self, split, index, role):
        if self.storage == "packed":
            if split not in self.readers:
                self.readers[split] = PackedTaskReader(self.directory / split, self.key)
            return self.readers[split].get(int(index), role)
        token = (split, int(index), role)
        if token in self.cache:
            self.hits += 1
            self.cache.move_to_end(token)
            return self.cache[token][0]
        sequence = FeatureSequence.load(self.directory / f"{split}_{index:04d}_{role}.npz", self.key)
        self.loads += 1
        size = sum(getattr(sequence, name).nbytes for name in _DTYPES if getattr(sequence, name) is not None)
        if size <= self.limit:
            while self.bytes + size > self.limit and self.cache:
                _, (_, removed) = self.cache.popitem(last=False)
                self.bytes -= removed
            self.cache[token] = (sequence, size)
            self.bytes += size
        return sequence

    def recipe(self, split, index):
        if self.storage == "packed":
            if split not in self.readers:
                self.readers[split] = PackedTaskReader(self.directory / split, self.key)
            return self.readers[split].record["recipes"][int(index)]
        return json.loads((self.directory / f"{split}_{index:04d}.json").read_text(encoding="utf-8"))

    def close(self):
        self.cache.clear()
        self.bytes = 0
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
