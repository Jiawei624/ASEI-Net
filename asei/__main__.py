from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import traceback

import numpy as np
import pandas as pd

from .config import (DEFAULT_CONFIG, configure_cuda, load_config, log_path, project_path,
                     report_path, resolve_paths, write_json)
from .data import discover
from .preparation import inspect_entities, save_prepared_manifest, training_manifest


def main():
    parser = argparse.ArgumentParser(description="ASEI-Net normal-only training and causal anomaly detection")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", "--config-path", default=str(DEFAULT_CONFIG))
        p.add_argument("--data-root", help="Override configured data.root")
        p.add_argument("--manifest", help="Override configured source manifest")
        p.add_argument("--datasets", "--data-set-name", nargs="+", help="Dataset names; otherwise use data.datasets")
        p.add_argument("--entities", "--data-name-list", nargs="+")
        if name == "run":
            p.add_argument("--output", help="Override runtime.output")
            p.add_argument("--seeds", nargs="+", type=int, help="Override runtime.seeds")
            p.add_argument("--device", help="Override logical runtime.device")
            p.add_argument("--stage", choices=["all", "representation", "cache", "detector", "calibrate", "evaluate"], default="all")
            p.add_argument("--variant", choices=["full", "mlp", "single_gru", "no_alignment", "zero_alignment", "fixed_gate", "independent_gru"])
            p.add_argument("--sources", nargs="*", choices=["relation", "marginal", "joint"])
            p.add_argument("--skip-vus", action="store_true")
            p.add_argument("--all-tab-label-metrics", action="store_true")
            p.add_argument("--diagnostics", action="store_true")
            p.add_argument("--log", action="store_true", help="Append stdout/stderr to the configured runtime log file")
        else:
            p.add_argument("--output", help="Override reports.inventory; the training output root is unchanged")
            p.add_argument("--read-data", action="store_true", help="Validate normality, channel schema and source hashes")
            p.add_argument("--prepare", action="store_true", help="Read data and save eligible manifest at its configured path")
    p = sub.add_parser("aggregate")
    p.add_argument("--config", "--config-path", default=str(DEFAULT_CONFIG))
    p.add_argument("--root", help="Override configured runtime.output")
    p.add_argument("--output", help="Override configured reports.summary")
    p.add_argument("--baseline-root")
    p.add_argument("--metric", default="score.auc_pr")
    p.add_argument("--variant", default="full")
    p.add_argument("--baseline-variant", default="single_gru")
    p = sub.add_parser("predict")
    p.add_argument("--config", "--config-path", default=str(DEFAULT_CONFIG))
    p.add_argument("--bundle", required=True)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", help="Default: reports.predictions/<input stem>_alarms.csv")
    p.add_argument("--device", help="Override logical runtime.device")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "run" and args.log:
        target = log_path(cfg)
        target.parent.mkdir(parents=True, exist_ok=True)
        print("Log:", target, flush=True)
        with open(target, "a", encoding="utf-8", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    execute(args, cfg)
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1)
    else:
        execute(args, cfg)


def execute(args, cfg):
    if args.command == "aggregate":
        from .aggregate import aggregate, paired_comparison
        root = project_path(args.root) if args.root else project_path(cfg["runtime"]["output"])
        output_path = project_path(args.output) if args.output else report_path(cfg, "summary", root)
        output = aggregate(root, output_path)
        if args.baseline_root:
            result = paired_comparison(root, project_path(args.baseline_root), args.metric, args.variant, args.baseline_variant)
            write_json(output / "paired_comparison.json", result)
        print(output)
        return
    if args.command == "predict":
        cfg["runtime"]["device"] = args.device or cfg["runtime"]["device"]
        configure_cuda(cfg)
        from .pipeline import load_bundle
        source = project_path(args.input_csv)
        target = project_path(args.output) if args.output else report_path(cfg, "predictions") / (source.stem + "_alarms.csv")
        stream, names = load_bundle(project_path(args.bundle), cfg["runtime"]["device"])
        frame = pd.read_csv(source, usecols=names)
        output = [stream.push(row) for row in frame[names].to_numpy(np.float64)]
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(output).to_csv(target, index=False)
        print(target)
        return
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.command == "run" and args.output:
        cfg["runtime"]["output"] = args.output
    cfg = resolve_paths(cfg)
    datasets = set(ds.upper() for ds in (args.datasets or cfg["data"]["datasets"]))
    manifest = training_manifest(cfg, datasets, args.entities) if args.command == "run" else cfg["data"].get("manifest")
    entities = [e for e in discover(cfg["data"]["root"], manifest,
                                    multivariate_only=cfg["data"].get("multivariate_only", True))
                if e.dataset in datasets and (not args.entities or e.name in args.entities or e.path.name in args.entities)]
    if not entities:
        raise SystemExit("No entities match the requested datasets/entities. See the configured inventory report.")
    if args.command == "inspect":
        frame = inspect_entities(entities, cfg, read_data=args.read_data or args.prepare)
        target = project_path(args.output) if args.output else report_path(cfg, "inventory")
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        print(frame.to_string(index=False))
        print("Inventory:", target)
        if args.prepare:
            manifest_path = save_prepared_manifest(entities, frame, cfg, datasets, args.entities)
            print("Eligible manifest:", manifest_path)
            if not frame.status.eq("eligible").any():
                raise SystemExit("No eligible entity. See the inventory reasons before training.")
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
    configure_cuda(cfg)
    # CUDA visibility must be configured before importing PyTorch through the pipeline.
    from .pipeline import run_entity
    seeds = args.seeds if args.seeds is not None else cfg["runtime"].get("seeds", list(range(5)))
    for entity in entities:
        for seed in seeds:
            run_entity(entity, cfg, seed, args.stage, args.diagnostics)


if __name__ == "__main__":
    main()
