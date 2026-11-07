"""Plain-text progress messages limited by elapsed time."""
from __future__ import annotations

import math
import time

from .resources import cooperate


class Progress:
    """Timed block-level progress, with explicit phase transitions."""
    def __init__(self, description, total, interval_seconds=30):
        self.description, self.total = description, total
        self.interval = float(interval_seconds)
        self.began = self.last = time.perf_counter()
        print(f"{description} start total={total}", flush=True)

    def update(self, completed, force=False):
        now = time.perf_counter()
        if force or completed == self.total or (self.interval and now - self.last >= self.interval):
            elapsed = now - self.began
            remaining = elapsed * (self.total - completed) / max(1, completed)
            print(f"{self.description} progress={completed}/{self.total} "
                  f"elapsed={elapsed:.1f}s remaining={remaining:.1f}s", flush=True)
            self.last = now


def periodic_progress(iterable, description, interval_seconds=0):
    """Yield items unchanged; callers report completion or an epoch summary."""
    interval = float(interval_seconds)
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("log_interval_seconds must be a finite nonnegative number.")
    if interval == 0:
        for item in iterable:
            cooperate()
            yield item
        return
    total = len(iterable)
    began = last = time.perf_counter()
    for completed, item in enumerate(iterable, 1):
        cooperate()
        yield item
        now = time.perf_counter()
        if completed < total and now - last >= interval:
            print(f"{description} progress={completed}/{total} ({100 * completed / total:.1f}%) "
                  f"elapsed={now - began:.1f}s", flush=True)
            last = now
