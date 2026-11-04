import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from asei.config import ROOT, load_config, resolve_config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(5)


@pytest.fixture
def tiny_config():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["model"].update(window=32, d_model=16, heads=4, layers=1, dropout=0.0,
                         marginal_prototypes=4, joint_prototypes=8)
    cfg["stage1"].update(epochs=1, batch_size=32, stride=8, warmup_epochs=1)
    cfg["stage2"].update(epochs=1, batch_pairs=2, block_length=32, train_pairs=8, validation_pairs=8)
    cfg["evaluation"].update(score_metrics=["auc_roc", "auc_pr"], latency_points=3)
    cfg["runtime"].update(device="cpu", threads=1, feature_batch_size=64)
    return resolve_config(cfg, 320, 3)
