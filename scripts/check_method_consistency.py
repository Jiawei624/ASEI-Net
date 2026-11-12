"""Check that dataset profiles differ only in capacity and training settings."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asei.config import fingerprint, implementation_fingerprint, load_config, resolve_config, write_json

DATASETS = ("psm", "pump", "smd", "swat")
CAPACITY = {"d_model", "heads", "layers", "dropout", "marginal_prototypes", "joint_prototypes"}
STAGE1_TRAINING = {"epochs", "patience", "batch_size", "lr", "weight_decay", "stride",
                   "max_train_points", "max_validation_points", "lr_patience", "lr_factor", "min_lr"}
TEACHER_TRAINING = {"epochs", "patience", "lr", "weight_decay", "batch_sequences", "train_sequences",
                    "lr_patience", "lr_factor", "min_lr"}
DETECTOR_TRAINING = {"epochs", "patience", "batch_pairs", "lr", "train_pairs", "validation_pairs",
    "weight_decay", "lr_patience", "lr_factor", "min_lr", "evidence_hidden", "readout_hidden",
    "lambda_state", "lambda_tail", "lambda_pair", "lambda_channel", "onset_fraction", "hard_negative_fraction",
    "selection_normal_points", "selection_fpr_limit", "selection_coverage_weight"}


def method_view(cfg):
    """Keep all method-defining options, including whether each loss is enabled."""
    cfg = copy.deepcopy(cfg)
    def without(section, allowed):
        return {key: value for key, value in cfg[section].items() if key not in allowed}
    return dict(
        data=without("data", {"datasets", "root", "manifest", "archive"}),
        model=without("model", CAPACITY | {"channels"}),
        memory=without("memory", {"pool_size", "rare_fraction"}),
        stage1=without("stage1", STAGE1_TRAINING),
        teacher=without("teacher", TEACHER_TRAINING),
        stage2=without("stage2", DETECTOR_TRAINING),
        enabled_losses={name: cfg["stage2"]["lambda_" + name] > 0 for name in ("state", "tail", "pair", "channel")},
        calibration=cfg["calibration"], evaluation=cfg["evaluation"],
        execution=dict(detector=cfg["runtime"].get("detector_execution", "block"),
                       reference=cfg["runtime"].get("reference_execution", "auto")))


def differences(left, right, prefix=""):
    result = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            result.extend(differences(a, b, prefix + key + "."))
        elif a != b:
            result.append(prefix + key)
    return result


def check_profile(cfg, recipe):
    paths = differences(method_view(recipe), method_view(cfg))
    if paths:
        raise ValueError("Full-method definition differs from method.yaml: " + ", ".join(paths))
    if not all(method_view(cfg)["enabled_losses"].values()):
        raise ValueError("All original loss components must be enabled in a full-method profile.")


def check_profiles(config_root=None):
    config_root = Path(config_root or ROOT / "configs").resolve()
    recipe = resolve_config(load_config(config_root / "method.yaml"), 10000, 4)
    profiles = []
    for name in DATASETS:
        cfg = resolve_config(load_config(config_root / (name + ".yaml")), 10000, 4)
        check_profile(cfg, recipe)
        profiles.append(dict(dataset=name.upper(), method_key=fingerprint(method_view(cfg)),
            model={key: cfg["model"][key] for key in sorted(CAPACITY)},
            detector={key: cfg["stage2"][key] for key in ("evidence_hidden", "readout_hidden", "epochs", "batch_pairs", "lr")}))
    assert len({p["method_key"] for p in profiles}) == 1
    return dict(shared_method=method_view(recipe), method_key=profiles[0]["method_key"], profiles=profiles,
                implementation_key=implementation_fingerprint(), real_dataset_training=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    report = check_profiles(args.config_root)
    if args.output:
        write_json(args.output, report)
    print("PASS: PSM/PUMP/SMD/SWAT share architecture, loss components, preprocessing and calibration.")
    print("Shared method:", report["method_key"])
    for item in report["profiles"]:
        print(item["dataset"], item["detector"])


if __name__ == "__main__":
    main()
