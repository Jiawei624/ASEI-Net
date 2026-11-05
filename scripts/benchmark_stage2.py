"""Measure the detector's forward/backward throughput without dataset training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asei.config import DEFAULT_CONFIG, configure_cuda, load_config, project_path, write_json
from asei.resources import configure_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--config-path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    device_args = parser.add_mutually_exclusive_group()
    device_args.add_argument("--gpu", "--gpus", type=int, help="Physical GPU index")
    device_args.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_cuda(cfg, gpu=args.gpu, device=args.device)
    configure_cpu(cfg)
    import torch
    from asei.model import DualTrajectoryDetector
    from asei.pipeline import get_device

    device = get_device(cfg["runtime"]["device"])
    pairs = args.pairs if args.pairs is not None else cfg["stage2"]["batch_pairs"]
    length = args.length if args.length is not None else cfg["stage2"]["block_length"]
    if min(args.channels, pairs, length, args.steps) < 1 or args.warmup < 0:
        raise ValueError("Benchmark dimensions and steps must be positive; warmup must be nonnegative.")
    configure_cpu(cfg)
    torch.manual_seed(42)
    batch, channels = 2 * pairs, args.channels
    identity_dim = cfg["model"]["d_model"]
    identity_dim = 96 if identity_dim == "auto" else identity_dim
    model = DualTrajectoryDetector(torch.randn(channels, identity_dim, device=device),
                                   variant=cfg["stage2"]["variant"], sources=cfg["stage2"]["sources"],
                                   fixed_retention=cfg["stage2"]["fixed_retention"]).to(device).train()
    shape = (batch, length, channels)
    f = {name: torch.randn(*shape, 37, device=device) for name in ("chi", "marginal", "joint")}
    f.update(prediction=torch.randn(*shape, 2, device=device), residual=torch.randn(*shape, 2, device=device))
    f.update({name: torch.ones(*shape, dtype=torch.bool, device=device)
              for name in ("m_valid", "j_valid", "observed", "prediction_ready")})

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(execution):
        model.zero_grad(set_to_none=True)
        if execution == "block":
            output, _ = model.forward_block(f)
            scores = output["score"]
        else:
            state, outputs = None, []
            for t in range(length):
                output, state = model.step({k: v[:, t] for k, v in f.items()}, state)
                outputs.append(output["score"])
            scores = torch.stack(outputs, 1)
        scores.square().mean().backward()
        return scores.detach()

    reference = run("step")
    optimized = run("block")
    tf32 = device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32
    torch.testing.assert_close(reference, optimized, atol=3e-4 if tf32 else 3e-5, rtol=3e-3 if tf32 else 3e-4)
    max_difference = float((reference - optimized).abs().max().cpu())
    del reference, optimized
    results = {}
    for execution in ("step", "block"):
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
        results[execution] = {"seconds_per_batch": seconds / args.steps,
            "sequence_points_per_second": batch * length * args.steps / seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    results.update(speedup=results["step"]["seconds_per_batch"] / results["block"]["seconds_per_batch"],
                   device=str(device), torch=str(torch.__version__), pairs=pairs, channels=channels, length=length,
                   max_score_difference=max_difference, allow_tf32=bool(tf32),
                   accelerator=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   scope="Synthetic detector forward/backward only; excludes data loading and frozen feature extraction.")
    output = (project_path(args.output) if args.output else Path(cfg["runtime"]["output"]) / "benchmarks" /
              (Path(args.config).stem + "_stage2.json"))
    write_json(output, results)
    print(results)
    print("Benchmark report:", output)


if __name__ == "__main__":
    main()
