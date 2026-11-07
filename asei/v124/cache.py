"""Packed raw tasks and frozen forecast errors, bounded RAM and resumable writes."""
from __future__ import annotations

import json
from pathlib import Path
from collections import OrderedDict

import numpy as np

from ..data import Record
from ..config import write_json
from ..tasks import event_ranges


class PairCache124:
    FIELDS = {"z": (np.float32, 1), "mask": (np.uint8, 1), "error": (np.float32, 2), "ready": (np.uint8, 2)}

    def __init__(self, directory, key, channels, memory_mb=256):
        self.directory, self.key, self.channels = Path(directory), key, channels
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "index.json"
        self.index = {"format": "in1_2_4/raw-pairs/1", "key": key, "channels": channels, "entries": []}
        if self.path.exists():
            self.index = json.loads(self.path.read_text(encoding="utf-8"))
            if self.index.get("format") != "in1_2_4/raw-pairs/1" or self.index["key"] != key or self.index["channels"] != channels:
                raise ValueError("Task cache identity differs; use a new run.")
        self.maps, self.lru, self.bytes = {}, OrderedDict(), 0
        self.limit = max(0, int(memory_mb * 1024 ** 2))
        self.offsets = np.r_[0, np.cumsum([e["length"] for e in self.index["entries"]], dtype=np.int64)].tolist()
        self._recover_uncommitted_suffix()

    def _file(self, role, field):
        return self.directory / (role + "_" + field + ".bin")

    def _recover_uncommitted_suffix(self):
        rows = sum(e["length"] for e in self.index["entries"])
        for role in ("clean", "corrupt"):
            for field, (dtype, width) in self.FIELDS.items():
                path = self._file(role, field)
                expected = rows * self.channels * width * np.dtype(dtype).itemsize
                if path.exists() and path.stat().st_size < expected:
                    raise ValueError("Truncated committed task cache: " + str(path))
                if path.exists() and path.stat().st_size > expected:
                    with path.open("r+b") as stream:
                        stream.truncate(expected)
                if expected and not path.exists():
                    raise ValueError("Missing committed task cache: " + str(path))

    def append(self, pair, errors, ready):
        if self.maps:
            raise RuntimeError("Close read maps before appending task records.")
        if pair.clean.z.shape != pair.corrupt.z.shape:
            raise ValueError("Paired task shapes differ.")
        for role, record, error, mask in zip(("clean", "corrupt"), (pair.clean, pair.corrupt), errors, ready):
            values = dict(z=record.z, mask=record.mask, error=error, ready=mask)
            for field, (dtype, _) in self.FIELDS.items():
                with self._file(role, field).open("ab") as stream:
                    np.asarray(values[field], dtype=dtype).tofile(stream)
        entry = dict(length=len(pair.clean.z), target_start=pair.clean.target_start, recipe=pair.recipe,
                     label_ranges=event_ranges(pair.labels))
        self.index["entries"].append(entry)
        self.offsets.append(self.offsets[-1] + entry["length"])
        temporary = self.path.with_suffix(".tmp")
        write_json(temporary, self.index)
        temporary.replace(self.path)

    def get(self, index):
        if index in self.lru:
            value, size = self.lru.pop(index)
            self.lru[index] = (value, size)
            return value
        entry = self.index["entries"][index]
        start = self.offsets[index]
        length = entry["length"]
        result = []
        size = 0
        for role in ("clean", "corrupt"):
            fields = {}
            for field, (dtype, width) in self.FIELDS.items():
                identity = role, field
                if identity not in self.maps:
                    self.maps[identity] = np.memmap(self._file(role, field), dtype=dtype, mode="r")
                begin, end = start * self.channels * width, (start + length) * self.channels * width
                shape = (length, self.channels) if width == 1 else (length, self.channels, width)
                fields[field] = np.array(self.maps[identity][begin:end].reshape(shape), copy=True)
                size += fields[field].nbytes
            result.append((Record(fields["z"], fields["mask"].astype(bool), entry["target_start"]),
                           fields["error"], fields["ready"].astype(bool)))
        labels = np.zeros(length, bool)
        for start, stop in entry["label_ranges"]:
            labels[start:stop] = True
        value = (*result, labels, entry["recipe"])
        if size <= self.limit:
            while self.lru and self.bytes + size > self.limit:
                _, (_, old_size) = self.lru.popitem(last=False)
                self.bytes -= old_size
            self.lru[index] = (value, size)
            self.bytes += size
        return value

    def close(self):
        self.lru.clear()
        self.bytes = 0
        for array in self.maps.values():
            array._mmap.close()
        self.maps.clear()

    def __len__(self):
        return len(self.index["entries"])
