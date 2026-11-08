"""Bound this process tree's CPU use and yield on a busy shared Linux host."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time

from threadpoolctl import threadpool_limits
from .progress import log


def validate_cpu_config(runtime):
    for key, default in (("cpu_cores", 8), ("threads", 4), ("interop_threads", 1)):
        value = runtime.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("runtime." + key + " must be a positive integer.")
        runtime[key] = value
    for key in ("preparation_workers", "prefetch_batches"):
        value = runtime.get(key, 2)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("runtime." + key + " must be a nonnegative integer.")
        runtime[key] = value
    for key, default, maximum in (("cpu_max_percent", 80, 100),
                                   ("cpu_check_interval_seconds", 5, None)):
        try:
            value = float(runtime.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid runtime." + key) from exc
        if not math.isfinite(value) or value <= 0 or (maximum is not None and value > maximum):
            raise ValueError("Invalid runtime." + key)
        runtime[key] = value


def _allowed_cpus():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def _set_affinity(cpus):
    if not hasattr(os, "sched_setaffinity"):
        return False
    # Existing native thread pools also need the same mask. Future threads and
    # DataLoader children inherit it from their parent thread.
    os.sched_setaffinity(0, cpus)
    for entry in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(entry.name), cpus)
        except ProcessLookupError:
            pass  # A native thread can exit during enumeration.
    return True


def _cpu_counters():
    try:
        with open("/proc/stat", encoding="ascii") as stream:
            line = stream.readline().split()
        if line[0] != "cpu" or len(line) < 5:
            return None
        fields = [int(v) for v in line[1:9]]
        # guest/guest_nice, when present, are already included in user/nice.
        return sum(fields), fields[3] + (fields[4] if len(fields) > 4 else 0)
    except (OSError, ValueError, IndexError):
        return None


class CpuGovernor:
    def __init__(self, maximum, interval):
        self.maximum, self.interval = maximum, interval
        self.previous = _cpu_counters()
        self.last_check = time.monotonic()
        self.last_notice = -math.inf

    def cooperate(self):
        if self.previous is None or time.monotonic() - self.last_check < self.interval:
            return
        current = _cpu_counters()
        now = time.monotonic()
        previous, self.previous = self.previous, current
        self.last_check = now
        if current is None:
            return
        total, idle = current[0] - previous[0], current[1] - previous[1]
        busy = 100 * (1 - min(total, max(0, idle)) / total) if total > 0 else 0
        if busy > self.maximum:
            if now - self.last_notice >= 300:
                log(f"CPU busy={busy:.1f}% exceeds {self.maximum:g}%; yielding briefly.")
                self.last_notice = now
            # Other users' sustained load must never stop this process indefinitely.
            time.sleep(min(self.interval, .25))


_governor = None
_governor_settings = None
_last_report = None
_affinity_source = None
_applied_affinity = None
_native_limits = None


def configure_cpu(cfg):
    """Call before NumPy/PyTorch imports; safe to repeat for sequential seeds."""
    global _governor, _governor_settings, _last_report, _affinity_source, _applied_affinity, _native_limits
    runtime = cfg["runtime"]
    validate_cpu_config(runtime)
    current_cpus = _allowed_cpus()
    if _applied_affinity is None or current_cpus != _applied_affinity:
        _affinity_source = current_cpus
    available = _affinity_source
    budget = min(runtime["cpu_cores"], max(1, int(len(available) * runtime["cpu_max_percent"] / 100)))
    # Separate GPU jobs get separate CPU sets when the host has enough CPUs.
    visible = str(runtime.get("cuda_visible_devices", "0"))
    gpu = int(visible) if runtime.get("device") != "cpu" and visible.isdigit() else 0
    offset = (gpu * budget) % len(available)
    cpus = (available + available)[offset:offset + budget]
    affinity = _set_affinity(cpus)
    _applied_affinity = sorted(cpus) if affinity else None
    threads = min(runtime["threads"], budget)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    # NumPy may already be imported by embedded callers or configuration loading.
    # Environment variables alone do not resize existing BLAS/OpenMP pools.
    _native_limits = threadpool_limits(limits=threads)
    numexpr = sys.modules.get("numexpr")
    if numexpr is not None:
        numexpr.set_num_threads(threads)
    torch = sys.modules.get("torch")
    if torch is not None:
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != runtime["interop_threads"]:
            try:
                torch.set_num_interop_threads(runtime["interop_threads"])
            except RuntimeError:
                # Embedded callers may already have started the interop pool;
                # Linux affinity still bounds its CPU consumption.
                pass
    workers = min(cfg.get("stage1", {}).get("workers", 0), max(0, budget - threads))
    report = {"host_logical_cpus": os.cpu_count(), "cpu_budget": budget,
              "cpu_affinity": cpus if affinity else None, "threads": threads,
              "interop_threads": torch.get_num_interop_threads() if torch is not None else runtime["interop_threads"],
              "workers": workers, "cpu_max_percent": runtime["cpu_max_percent"],
              "cpu_check_interval_seconds": runtime["cpu_check_interval_seconds"]}
    report["preparation_workers"] = min(runtime.get("preparation_workers", 2), max(0, budget - threads))
    report["prefetch_batches"] = runtime.get("prefetch_batches", 2) if report["preparation_workers"] else 0
    runtime["effective_cpu"] = report
    settings = (runtime["cpu_max_percent"], runtime["cpu_check_interval_seconds"])
    if settings != _governor_settings:
        _governor = CpuGovernor(*settings)
        _governor_settings = settings
    report["system_load_guard"] = _governor.previous is not None
    if report != _last_report:
        log(f"CPU budget={budget} threads={threads} workers={workers} "
            f"prepare_workers={report['preparation_workers']} prefetch={report['prefetch_batches']} "
            f"affinity={report['cpu_affinity']} host_limit={settings[0]:g}% "
            f"load_guard={report['system_load_guard']}", cfg)
        _last_report = report.copy()
    return report


def cooperate():
    if _governor is not None:
        _governor.cooperate()
