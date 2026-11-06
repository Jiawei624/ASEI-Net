from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import traceback
import json

from .config import (DEFAULT_CONFIG, configure_cuda, load_config, project_path,
                     report_path, resolve_paths, write_json)
from .resources import configure_cpu


def main():
    parser = argparse.ArgumentParser(description="ASEI-Net normal-only training and causal anomaly detection")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "run"):
        p = sub.add_parser(name)
        p.add_argument("--config", "--config-path", help="YAML/JSON configuration; resume defaults to the saved run configuration")
        p.add_argument("--data-root", help="Override configured data.root")
        p.add_argument("--manifest", help="Override configured source manifest")
        p.add_argument("--datasets", "--data-set-name", nargs="+", help="Dataset names; otherwise use data.datasets")
        p.add_argument("--entities", "--data-name-list", nargs="+")
        if name == "run":
            p.add_argument("--output", help="Override runtime.output")
            p.add_argument("--resume", help="Reuse a previous run ID; legacy reuses output directories created before timestamps")
            p.add_argument("--seeds", "--seed", nargs="+", type=int, help="Override runtime.seeds (one or more seeds)")
            device = p.add_mutually_exclusive_group()
            device.add_argument("--gpu", "--gpus", type=int, help="Physical GPU index, e.g. 0 or 1; uses logical cuda:0")
            device.add_argument("--device", help="Override logical runtime.device; use cpu for CPU execution")
            p.add_argument("--stage", choices=["all", "representation", "cache", "detector", "calibrate", "evaluate"], default="all")
            p.add_argument("--variant", choices=["full", "mlp", "single_gru", "no_alignment", "zero_alignment", "fixed_gate", "independent_gru"])
            p.add_argument("--sources", nargs="*", choices=["relation", "marginal", "joint"])
            p.add_argument("--skip-vus", action="store_true")
            p.add_argument("--keep-intermediates", action="store_true", help="Keep task caches, predictions and diagnostic files after success")
            p.add_argument("--all-label-metrics", "--all-tab-label-metrics", action="store_true")
            p.add_argument("--all-metrics", action="store_true", help="Compute all 16 label and 6 score detection metrics (the default)")
            p.add_argument("--diagnostics", action="store_true")
            p.add_argument("--log", action="store_true", help="Append training output to the configured log filename with its run timestamp")
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
    p.add_argument("--seeds", "--seed", nargs="+", type=int, help="Seeds to aggregate; otherwise use runtime.seeds")
    p = sub.add_parser("export")
    p.add_argument("--root", required=True, help="Result JSON/CSV file or experiment directory")
    p.add_argument("--output", required=True, help="Destination CSV containing the 22 detection metrics")
    p.add_argument("--recompute-labels", action="store_true", help="Recompute label metrics from original predictions.npz, reusing existing score metrics")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p = sub.add_parser("predict")
    p.add_argument("--config", "--config-path", default=str(DEFAULT_CONFIG))
    p.add_argument("--bundle", required=True)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", help="Default: reports.predictions/<input stem>_alarms.csv")
    device = p.add_mutually_exclusive_group()
    device.add_argument("--gpu", "--gpus", type=int, help="Physical GPU index, e.g. 0 or 1")
    device.add_argument("--device", help="Override logical runtime.device")
    args = parser.parse_args()
    cfg = load_config(args.config or DEFAULT_CONFIG)
    if args.command == "run" and args.resume and args.resume != "legacy" and args.config is None:
        from pathlib import Path
        root = project_path(args.output or cfg["runtime"]["output"])
        # start_run performs the authoritative run-ID/path validation.
        from .runs import start_run
        import copy
        lookup = copy.deepcopy(cfg)
        lookup["runtime"]["output"] = str(root)
        directory = start_run(lookup, resume=args.resume)
        saved = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
        cfg = resolve_paths(saved["config"])
    execute(args, cfg)


def execute(args, cfg):
    if args.command == "export":
        configure_cpu(cfg)
        from .results import export_results
        print(export_results(project_path(args.root), project_path(args.output), args.recompute_labels))
        return
    if args.command in {"run", "predict"}:
        configure_cuda(cfg, gpu=args.gpu, device=args.device)
    configure_cpu(cfg)
    if args.command == "aggregate":
        from .aggregate import aggregate, paired_comparison
        from .runs import new_directory
        root = project_path(args.root) if args.root else project_path(cfg["runtime"]["output"])
        output_path = project_path(args.output) if args.output else new_directory(report_path(cfg, "summary", root), cfg)
        selected_seeds = args.seeds if args.seeds is not None else cfg["runtime"].get("seeds", [41, 42, 43, 44, 45])
        output = aggregate(root, output_path, seeds=selected_seeds)
        if args.baseline_root:
            result = paired_comparison(root, project_path(args.baseline_root), args.metric, args.variant, args.baseline_variant,
                                       seeds=selected_seeds)
            write_json(output / "paired_comparison.json", result)
        print(output)
        return
    if args.command == "predict":
        import numpy as np
        import pandas as pd
        from .pipeline import load_bundle
        configure_cpu(cfg)  # Apply thread limits after importing PyTorch, too.
        source = project_path(args.input_csv)
        target = project_path(args.output) if args.output else report_path(cfg, "predictions") / (source.stem + "_alarms.csv")
        stream, names = load_bundle(project_path(args.bundle), cfg["runtime"]["device"])
        frame = pd.read_csv(source, usecols=names)
        output = [stream.push(row) for row in frame[names].to_numpy(np.float64)]
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(output).to_csv(target, index=False)
        print(target)
        return
    from .data import DATASETS, discover
    from .preparation import inspect_entities, save_prepared_manifest, training_manifest
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.command == "run" and args.output:
        cfg["runtime"]["output"] = args.output
    cfg = resolve_paths(cfg)
    datasets = set(ds.upper() for ds in (args.datasets or cfg["data"]["datasets"]))
    unsupported = datasets - set(DATASETS)
    if unsupported:
        raise SystemExit("Unsupported datasets: " + ", ".join(sorted(unsupported)) + ". Available: " + ", ".join(DATASETS))
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
    if args.variant:
        cfg["stage2"]["variant"] = args.variant
    if args.sources is not None:
        cfg["stage2"]["sources"] = args.sources
    cfg["evaluation"]["skip_vus"] = args.skip_vus
    if args.keep_intermediates or args.diagnostics:
        cfg["runtime"]["keep_intermediates"] = True
    seeds = args.seeds if args.seeds is not None else cfg["runtime"].get("seeds", [41, 42, 43, 44, 45])
    if args.stage not in {"all", "representation"} and not args.resume:
        raise ValueError("--stage " + args.stage + " requires --resume <run_id>; use --resume legacy for older results.")
    from .runs import start_run, run_log_path
    cfg["runtime"]["seeds"] = list(seeds)
    cfg["data"]["datasets"] = sorted(datasets)
    directory = start_run(cfg, resume=args.resume)
    print("Run ID:", cfg["runtime"]["run_id"], flush=True)
    print("Run directory:", directory, flush=True)

    def train():
        # CUDA visibility was configured before importing PyTorch through the pipeline.
        from .pipeline import run_entity
        if args.log:
            print("Run ID:", cfg["runtime"]["run_id"], "directory:", directory, flush=True)
        for entity in entities:
            for seed in seeds:
                run_entity(entity, cfg, seed, args.stage, args.diagnostics)

    if args.log:
        target = run_log_path(cfg)
        target.parent.mkdir(parents=True, exist_ok=True)
        print("Log:", target, flush=True)
        with open(target, "a", encoding="utf-8", buffering=1) as stream:
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    train()
                except Exception:
                    traceback.print_exc()
                    raise SystemExit(1)
    else:
        train()


if __name__ == "__main__":
    main()
