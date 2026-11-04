from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config, write_json
from .data import discover, load_entity
from .pipeline import run_entity, load_bundle
from .aggregate import aggregate, paired_comparison


def main():
    parser = argparse.ArgumentParser(description="ASEI-Net normal-only training and causal anomaly detection")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
        p.add_argument("--data-root")
        p.add_argument("--manifest")
        p.add_argument("--datasets", nargs="+")
        p.add_argument("--entities", nargs="+")
        p.add_argument("--output")
        if name == "run":
            p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
            p.add_argument("--device")
            p.add_argument("--stage", choices=["all", "representation", "cache", "detector", "calibrate", "evaluate"], default="all")
            p.add_argument("--variant", choices=["full", "mlp", "single_gru", "no_alignment", "zero_alignment", "fixed_gate", "independent_gru"])
            p.add_argument("--sources", nargs="*", choices=["relation", "marginal", "joint"])
            p.add_argument("--skip-vus", action="store_true")
            p.add_argument("--all-tab-label-metrics", action="store_true")
            p.add_argument("--diagnostics", action="store_true")
        else:
            p.add_argument("--read-data", action="store_true", help="Validate normality, channel schema and source hashes")
    p = sub.add_parser("aggregate")
    p.add_argument("--root", default="outputs")
    p.add_argument("--output", default="outputs/summary")
    p.add_argument("--baseline-root")
    p.add_argument("--metric", default="score.auc_pr")
    p.add_argument("--variant", default="full")
    p.add_argument("--baseline-variant", default="single_gru")
    p = sub.add_parser("predict")
    p.add_argument("--bundle", required=True)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "predict":
        stream, names = load_bundle(args.bundle, args.device)
        frame = pd.read_csv(args.input_csv, usecols=names)
        output = [stream.push(row) for row in frame[names].to_numpy(np.float64)]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(output).to_csv(args.output, index=False)
        print(args.output)
        return
    if args.command == "aggregate":
        output = aggregate(args.root, args.output)
        if args.baseline_root:
            result = paired_comparison(args.root, args.baseline_root, args.metric, args.variant, args.baseline_variant)
            write_json(output / "paired_comparison.json", result)
        print(output)
        return
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.output:
        cfg["runtime"]["output"] = args.output
    datasets = set(ds.upper() for ds in (args.datasets or cfg["data"]["datasets"]))
    entities = [e for e in discover(cfg["data"]["root"], cfg["data"].get("manifest"))
                if e.dataset in datasets and (not args.entities or e.name in args.entities)]
    if not entities:
        raise SystemExit("No entities match the requested datasets/entities.")
    if args.command == "inspect":
        rows = []
        for entity in entities:
            row = {"dataset": entity.dataset, "entity": entity.name, "file": str(entity.path),
                   "train_lens": entity.train_lens, "exists": entity.path.exists()}
            if args.read_data:
                try:
                    train, val, test, _, _, provenance = load_entity(entity, cfg["data"]["train_fraction"])
                    row.update(status="eligible", fit=len(train.z), validation=len(val.z), test=len(test.z),
                               channels=train.z.shape[1], source_sha256=provenance["source_sha256"])
                except (ValueError, FileNotFoundError, KeyError) as exc:
                    row.update(status="ineligible", reason=str(exc))
            rows.append(row)
        frame = pd.DataFrame(rows)
        print(frame.to_string(index=False))
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
        return
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.variant:
        cfg["stage2"]["variant"] = args.variant
    if args.sources is not None:
        cfg["stage2"]["sources"] = args.sources
    if args.skip_vus:
        cfg["evaluation"]["score_metrics"] = [m for m in cfg["evaluation"]["score_metrics"] if not m.startswith("VUS_")]
    if args.all_tab_label_metrics:
        cfg["evaluation"]["supplementary_label_metrics"] = ["adjust_accuracy", "adjust_precision", "adjust_recall", "adjust_f_score", "precision_at_k"]
    for entity in entities:
        for seed in args.seeds:
            run_entity(entity, cfg, seed, args.stage, args.diagnostics)


if __name__ == "__main__":
    main()
