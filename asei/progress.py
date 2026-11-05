"""Plain-text progress messages limited by elapsed time."""
from __future__ import annotations

import math
import time

from .resources import cooperate


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
