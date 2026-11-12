"""Bounded, ordered CPU preparation while the main thread owns all GPU work."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


def preparation_workers(cfg):
    runtime = cfg["runtime"]
    effective = runtime.get("effective_cpu", {})
    budget = effective.get("cpu_budget", runtime.get("cpu_cores", 8))
    native = min(budget, effective.get("threads", runtime.get("threads", 4)))
    return min(runtime.get("preparation_workers", 2), max(0, budget - native))


@contextmanager
def ordered_prefetch(function, items, workers=1, depth=2):
    """Preserve input order and exceptions; stop and join producers on early exit.

    The function must only prepare independent CPU inputs. It must not update a
    model, consume global random state, or access the CUDA stream. In particular,
    cache readers use one producer so their memory maps and LRU have one owner.
    """
    if workers <= 0 or depth <= 0:
        yield map(function, items)
        return
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asei-prepare")
    pending, source = deque(), iter(items)

    def submit():
        try:
            item = next(source)
        except StopIteration:
            return False
        pending.append(executor.submit(function, item))
        return True

    def results():
        for _ in range(depth):
            if not submit():
                break
        while pending:
            value = pending.popleft().result()
            submit()
            yield value

    try:
        yield results()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True)


def prefetch_depth(cfg):
    return cfg["runtime"].get("prefetch_batches", 2) if preparation_workers(cfg) else 0


@contextmanager
def background_writer(function, enabled=False):
    """One ordered disk writer, at most one queued group, with visible failures."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asei-write") if enabled else None
    pending = None
    def submit(value):
        nonlocal pending
        if executor is None:
            function(value)
        else:
            if pending is not None:
                pending.result()
            pending = executor.submit(function, value)
    try:
        yield submit
    finally:
        try:
            if pending is not None:
                pending.result()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
