"""Timestamped epoch summaries; inner loops remain silent."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time


def log(message, cfg=None):
    offset = (cfg or {}).get("runtime", {}).get("timestamp_utc_offset_hours", 8)
    stamp = datetime.now(timezone(timedelta(hours=float(offset)))).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{stamp}] {message}", flush=True)


def epoch_log(stage, number, total, train, validation, began, cfg, checkpoint, **details):
    seconds = time.perf_counter() - began
    elapsed = sum(entry.get("seconds", 0.) for entry in checkpoint.history)
    best = min(checkpoint.history, key=lambda entry: entry.get("validation_loss",
        entry.get("selection", {}).get("objective", entry.get("selection", {}).get("loss", float("inf")))))
    extra = " ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}" for key, value in details.items())
    stop = checkpoint.stale >= cfg[stage]["patience"]
    remaining = 0. if stop else elapsed / max(1, number) * (total - number)
    log(f"{stage} epoch={number}/{total} progress={100 * number / total:.1f}% "
        f"train={train:.6f} validation={validation:.6f} {extra} seconds={seconds:.1f} "
        f"elapsed={elapsed:.1f}s eta={remaining:.1f}s "
        f"best_epoch={best['epoch']} bad_epochs={checkpoint.stale} early_stop={stop}", cfg)


class Progress:
    """Compatibility counter for silent inner loops."""
    def __init__(self, description, total, interval_seconds=30):
        pass

    def update(self, completed, force=False):
        pass


def periodic_progress(iterable, description, interval_seconds=0):
    """Preserve CPU cooperation without emitting batch or block messages."""
    from .resources import cooperate
    for item in iterable:
        cooperate()
        yield item
