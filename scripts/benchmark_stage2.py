"""Measure causal detector forward/backward, including frozen reference replay."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asei.config import DEFAULT_CONFIG, configure_cuda, load_config, resolve_config, project_path, write_json
from asei.resources import configure_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--discrete-channels", type=int, default=0)
    parser.add_argument("--prepare-tasks", type=int, default=0, help="Also compare separate and batched independent task preparation")
    device_args = parser.add_mutually_exclusive_group()
    device_args.add_argument("--gpu", type=int)
    device_args.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_cuda(cfg, gpu=args.gpu, device=args.device)
    configure_cpu(cfg)
    import numpy as np
    import torch
    from asei.data import Record
    from asei.features import DescriptorSchema
    from asei.normal import NormalModel
    from asei.model import DualTrajectoryDetector
    from asei.pipeline import get_device

    configure_cpu(cfg)
    device = get_device(cfg["runtime"]["device"])
    pairs = args.pairs if args.pairs is not None else cfg["stage2"]["batch_pairs"]
    length = args.length if args.length is not None else cfg["stage2"]["block_length"]
    if min(args.channels, pairs, length, args.steps) < 1 or args.warmup < 0 or args.prepare_tasks < 0 or not 0 <= args.discrete_channels <= args.channels:
        raise ValueError("Invalid benchmark dimensions, steps or discrete-channel count.")
    window = cfg["model"]["window"]
    window = 128 if window == "auto" else window
    n = max(512, window + length + 64)
    cfg = resolve_config(cfg, n, args.channels)
    torch.manual_seed(42)
    values = np.random.default_rng(42).normal(size=(n, args.channels)).astype(np.float32)
    for i in range(args.discrete_channels):
        values[:, i] = (np.arange(n) // (7 + i)) % 3
    record = Record(values, np.ones_like(values, bool))
    normal = NormalModel(DescriptorSchema.fit(record).metadata, cfg).to(device)
    normal.memory.fit(record, pool_size=512)
    model = DualTrajectoryDetector(normal, cfg["stage2"]).to(device).train()
    batch = 2 * pairs
    raw = normal.schema.block([record] * batch, [0] * batch, length)
    cached_record = Record(values, record.mask, routing=model.routing(record, cfg["stage2"]["block_length"]))
    cached = normal.schema.block([cached_record] * batch, [0] * batch, length)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(execution):
        model.zero_grad(set_to_none=True)
        if execution == "step":
            state, outputs = None, []
            for t in range(length):
                out, state = model.step({k: v[:, t] for k, v in raw.items()}, state)
                outputs.append(out["dynamic"])
            scores = torch.stack(outputs, 1)
        else:
            scores = model.forward_block(cached if execution == "cached" else raw)[0]["dynamic"]
        scores.square().mean().backward()
        return scores.detach()

    reference = run("step")
    differences = {}
    for execution in ("block", "cached"):
        output = run(execution)
        torch.testing.assert_close(reference, output, atol=3e-4, rtol=3e-3)
        differences[execution] = float((reference - output).abs().max().cpu())
    results = {}
    for execution in ("step", "block", "cached"):
        for _ in range(args.warmup):
            run(execution)
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        began = time.perf_counter()
        for _ in range(args.steps):
            run(execution)
        synchronize()
        seconds = time.perf_counter() - began
        results[execution] = dict(seconds_per_batch=seconds / args.steps,
            sequence_points_per_second=batch * length * args.steps / seconds,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        print(f"{execution} seconds_per_batch={seconds / args.steps:.6f}", flush=True)
    results.update(device=str(device), torch=str(torch.__version__), pairs=pairs, channels=args.channels,
                   length=length, max_score_difference=differences,
                   cached_speedup=results["step"]["seconds_per_batch"] / results["cached"]["seconds_per_batch"],
                   scope="Synthetic forward/backward. Step/block include frozen retrieval; cached reuses exact routing. Excludes forecasting and disk I/O.")
    if args.prepare_tasks:
        tasks = []
        for i in range(args.prepare_tasks):
            task = values.copy()
            if args.discrete_channels < args.channels:
                task[80:120, -1] += .1 * i
            tasks.append(Record(task, record.mask.copy()))
        synchronize()
        began = time.perf_counter()
        separate = [model.routing(task, cfg["stage2"]["block_length"]) for task in tasks]
        synchronize()
        separate_seconds = time.perf_counter() - began
        began = time.perf_counter()
        together = model.routing_many(tasks, cfg["stage2"]["block_length"])
        synchronize()
        together_seconds = time.perf_counter() - began
        for first, second in zip(separate, together):
            for key in first:
                np.testing.assert_allclose(first[key], second[key], atol=3e-4, rtol=3e-4)
        results["task_preparation"] = dict(tasks=len(tasks), separate_seconds=separate_seconds,
            batched_seconds=together_seconds, speedup=separate_seconds / together_seconds,
            scope="Frozen routing only; excludes task synthesis, forecasting and disk writes.")
        print(f"task_preparation separate={separate_seconds:.6f}s batched={together_seconds:.6f}s", flush=True)
    target = project_path(args.output) if args.output else Path(cfg["runtime"]["output"]) / "benchmarks" / "stage2.json"
    write_json(target, results)
    print("Benchmark report:", target)


if __name__ == "__main__":
    main()
