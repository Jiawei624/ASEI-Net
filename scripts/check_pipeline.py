"""One-epoch synthetic smoke check of the complete new pipeline; no datasets needed."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--device", default=None, help="cpu or a logical CUDA device")
    device.add_argument("--gpu", type=int, help="Physical GPU index")
    parser.add_argument("--output", help="Keep smoke artifacts in this directory; default uses a temporary directory")
    args = parser.parse_args()
    from asei.config import load_config, configure_cuda
    cfg = load_config(ROOT / "configs" / "psm.yaml")
    configure_cuda(cfg, gpu=args.gpu, device=args.device or (None if args.gpu is not None else "cpu"))
    # Configure visibility before importing modules that import PyTorch.
    import numpy as np
    import pandas as pd
    from asei.data import Entity
    from asei.pipeline import run_entity, load_bundle

    def check(directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        t = np.arange(720)
        a = np.sin(t / 11) + rng.normal(0, .05, len(t))
        b = .7 * np.roll(a, 1) + rng.normal(0, .1, len(t))
        states = (t // 13) % 2
        labels = ((t >= 630) & (t < 680)).astype(int)
        a[labels > 0] += 1.5
        frame = pd.DataFrame(dict(sensor_a=a, sensor_b=b, actuator=states, label=labels))
        path = directory / "smoke.csv"
        frame.to_csv(path, index=False)
        cfg["runtime"].update(output=str(directory / "outputs"), keep_intermediates=True, seeds=[42])
        cfg["model"].update(window=32, d_model=8, heads=2, layers=1, dropout=0., marginal_prototypes=4, joint_prototypes=5)
        cfg["memory"].update(pool_size=64, prototype_chunk=2)
        cfg["stage1"].update(epochs=1, patience=1, batch_size=8, max_train_points=16, max_validation_points=16)
        cfg["teacher"].update(epochs=1, patience=1, train_sequences=1,
                               batch_sequences=1, sequence_length=16, block_length=16)
        cfg["stage2"].update(epochs=1, patience=1, train_pairs=4, validation_pairs=4, batch_pairs=2, block_length=32)
        cfg["evaluation"].update(skip_vus=True)
        destination = run_entity(Entity("SMD", "smoke", path, 600), cfg, 42, diagnostics=True)
        stream, names = load_bundle(destination / "bundle.pt", cfg["runtime"]["device"])
        output = stream.push_many(frame[names].to_numpy()[600:])
        with np.load(destination / "predictions.npz") as saved:
            np.testing.assert_allclose([r["p_mix"] for r in output], saved["p_mix"], atol=1e-6)
        print("PASS: forecast -> teacher -> pairs -> detector -> held-out calibration -> bundle -> causal inference")
        print("Smoke artifacts:", destination)

    if args.output:
        check(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="asei_smoke_") as directory:
            check(directory)


if __name__ == "__main__":
    main()
