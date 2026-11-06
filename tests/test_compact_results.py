import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from asei.config import load_config, resolve_config
from asei.data import Entity
from asei.metrics import (LABEL_METRICS, SCORE_METRICS, METRICS_PROTOCOL, evaluate, precision_at_k)
from asei.results import METRIC_COLUMNS, read_rows, export_results, prune_intermediates
from asei.compat import load_torch_checkpoint


@pytest.fixture(autouse=True)
def small_cpu(monkeypatch):
    from asei import resources
    torch.set_num_threads(1)
    monkeypatch.setattr(resources, "_governor", None)
    monkeypatch.setattr(resources, "_governor_settings", None)
    monkeypatch.setattr(resources, "_cpu_counters", lambda: None)
    monkeypatch.setattr(resources, "_set_affinity", lambda cpus: False)


def tiny_config(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg["model"].update(window=32, d_model=8, heads=2, layers=1, dropout=0.,
                         marginal_prototypes=4, joint_prototypes=4)
    cfg["stage1"].update(epochs=1, batch_size=32, stride=8, workers=0)
    cfg["stage2"].update(epochs=1, batch_pairs=2, block_length=32, train_pairs=4, validation_pairs=4)
    cfg["runtime"].update(output=str(tmp_path / "outputs"), device="cpu", threads=1,
                          feature_batch_size=64, keep_intermediates=False, log_interval_seconds=0)
    return cfg


def entity_file(tmp_path, dataset):
    frame = pd.DataFrame(np.random.default_rng(17).normal(size=(580, 3)), columns=["a", "b", "c"])
    frame["label"] = 0
    frame.loc[460:463, "label"] = 1
    path = tmp_path / (dataset + ".csv")
    frame.to_csv(path, index=False)
    return Entity(dataset, dataset, path, 400)


def test_topk_uses_scores_and_has_explicit_tie_semantics():
    y = np.array([0, 1, 0, 1])
    scores = np.array([.1, .9, .8, .2])
    cfg = load_config("configs/default.yaml")["evaluation"]
    values, alarms = evaluate(y, scores, np.ones(4, bool), 10, cfg)
    assert not alarms.any() and values["label"]["recall"] == 0
    assert values["label"]["precision_at_k"] == .5
    assert precision_at_k(y, np.ones(4)) == .5
    assert precision_at_k([1, 0, 1, 0], [2, 1, 1, 0]) == .75
    assert np.isnan(precision_at_k(np.zeros(4), scores))


def test_old_configs_are_normalized_to_twenty_two_detection_metrics(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["evaluation"].update(label_metrics=["f_score"], score_metrics=["auc_roc"], regression_metrics=["mse"])
    resolved = resolve_config(cfg, 1000, 3)
    assert set(resolved["evaluation"]["label_metrics"]) == LABEL_METRICS
    assert set(resolved["evaluation"]["score_metrics"]) == SCORE_METRICS
    assert resolved["evaluation"]["regression_metrics"] == []
    cfg["evaluation"]["skip_vus"] = True
    assert len(resolve_config(cfg, 1000, 3)["evaluation"]["score_metrics"]) == 4


def test_compact_two_dataset_run_has_one_csv_and_only_best_models_then_reuses_weights(tmp_path, monkeypatch):
    from asei import pipeline
    cfg = tiny_config(tmp_path)
    psm, pump = entity_file(tmp_path, "PSM"), entity_file(tmp_path, "PUMP")
    first = pipeline.run_entity(psm, cfg, 42)
    run_root = first.parents[4]
    run_id = run_root.name
    cfg["runtime"]["run_id"] = run_id
    second = pipeline.run_entity(pump, cfg, 42)
    assert first.parents[4] == second.parents[4]
    files = {p.relative_to(run_root).as_posix() for p in run_root.rglob("*") if p.is_file()}
    expected = {"run.json", "metrics.csv"} | {(p / "bundle.pt").relative_to(run_root).as_posix() for p in (first, second)}
    assert files == expected
    rows = read_rows(run_root / "metrics.csv")
    assert len(rows) == 2 and {row["dataset"] for row in rows} == {"PSM", "PUMP"}
    assert len(METRIC_COLUMNS) == 22
    assert all(row["metrics_protocol"] == METRICS_PROTOCOL and not row["missing_metrics"] for row in rows)
    before = load_torch_checkpoint(first / "bundle.pt", map_location="cpu")

    def forbidden(*args, **kwargs):
        raise AssertionError("A compact completed run must reuse its best networks")

    monkeypatch.setattr(pipeline, "fit_representation", forbidden)
    monkeypatch.setattr(pipeline, "fit_detector", forbidden)
    monkeypatch.setattr(pipeline, "build_task_cache", forbidden)
    pipeline.run_entity(psm, cfg, 42, stage="evaluate")
    after = load_torch_checkpoint(first / "bundle.pt", map_location="cpu")
    for group in ("representation", "detector"):
        for name in before[group]:
            assert torch.equal(before[group][name], after[group][name])
    assert before["experiment"]["result"]["metrics"] == after["experiment"]["result"]["metrics"]
    assert len(read_rows(run_root / "metrics.csv")) == 2
    pipeline.run_entity(psm, cfg, 42)
    stream, channels = pipeline.load_bundle(first / "bundle.pt", "cpu")
    assert len(channels) == 3
    assert not stream.push(np.zeros(3))["valid"]
    assert {p.relative_to(run_root).as_posix() for p in run_root.rglob("*") if p.is_file()} == expected


def test_export_preserves_missing_old_metrics_and_does_not_invent_topk(tmp_path):
    old = {"dataset": "SWAT", "entity": "SWAT", "seed": 42, "variant": "full", "run_id": "old",
           "metrics": {"label": {"accuracy": .7, "precision_at_k": 0}, "score": {"auc_pr": .3}}}
    (tmp_path / "swat.json").write_text(json.dumps(old), encoding="utf-8")
    output = export_results(tmp_path, tmp_path / "metrics_22.csv")
    row = read_rows(output)[0]
    assert row["accuracy"] == "0.7" and row["auc_pr"] == "0.3"
    assert row["precision_at_k"] == "" and "legacy_binary_input" in row["undefined_metrics"]
    assert row["adjust_f_score"] == "" and "adjust_f_score" in row["missing_metrics"]


def test_cleanup_rejects_paths_outside_selected_entity(tmp_path):
    base, other = tmp_path / "entity", tmp_path / "other"
    base.mkdir()
    other.mkdir()
    protected = other / "detector.pt"
    protected.write_bytes(b"keep")
    with pytest.raises(ValueError):
        prune_intermediates(base, other)
    assert protected.read_bytes() == b"keep"


def test_old_prediction_export_recomputes_labels_without_recomputing_score_metrics(tmp_path, monkeypatch):
    from asei import metrics as module
    y, scores = np.array([0, 1, 1, 0, 0, 1]), np.array([.1, .9, .2, .1, .3, .8])
    threshold = .5
    row = {"dataset": "SWAT", "entity": "SWAT", "seed": 42, "variant": "full", "run_id": "old",
           "calibration": {"threshold": threshold, "alpha": .01},
           "metrics": {"label": {"precision_at_k": 0}, "score": {name: .75 for name in SCORE_METRICS}}}
    (tmp_path / "result.json").write_text(json.dumps(row))
    np.savez(tmp_path / "predictions.npz", labels=y, scores=scores, valid=np.ones(6, bool), alarms=scores > threshold)
    original = module.score_metrics
    def no_ranking_recalculation(actual, scores, names, errors):
        assert names == []
        return original(actual, scores, names, errors)
    monkeypatch.setattr(module, "score_metrics", no_ranking_recalculation)
    csv_path = export_results(tmp_path, tmp_path / "labels_recomputed.csv", recompute_labels=True)
    result = read_rows(csv_path)[0]
    assert not result["missing_metrics"] and result["metrics_protocol"] == METRICS_PROTOCOL
    assert float(result["precision_at_k"]) == pytest.approx(2 / 3)
    assert float(result["adjust_recall"]) == 1 and float(result["VUS_PR"]) == .75
    assert json.loads((tmp_path / "result.json").read_text()) == row


def test_cli_resume_uses_saved_config_if_config_is_omitted(tmp_path, monkeypatch):
    from asei import __main__ as cli
    from asei.runs import start_run
    cfg = tiny_config(tmp_path)
    root = start_run(cfg)
    import sys
    monkeypatch.setattr(sys, "argv", ["asei", "run", "--output", cfg["runtime"]["output"],
                                    "--resume", root.name, "--datasets", "PSM", "--seed", "42", "--gpu", "0"])
    seen = []
    monkeypatch.setattr(cli, "execute", lambda args, loaded: seen.append(loaded))
    cli.main()
    assert seen[0]["model"]["window"] == 32 and seen[0]["stage1"]["stride"] == 8
