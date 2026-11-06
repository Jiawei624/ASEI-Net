import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from asei.compat import load_torch_checkpoint
from asei.config import fingerprint, load_config
from asei.data import Entity, Record
from asei.features import FeatureSequence
from asei.tasks import (TaskPair, observable_positive_mask, phase_weights,
                        TASK_SUPERVISION_PROTOCOL)


def test_unchanged_event_prefix_has_no_positive_loss_but_keeps_event_boundary():
    clean = Record(np.zeros((32, 2), np.float32), np.ones((32, 2), bool))
    corrupt = Record(clean.z.copy(), clean.mask.copy())
    corrupt.z[8:10, 0] = 1
    labels = np.zeros(32, np.int8)
    labels[4:22] = 1
    original_labels = labels.copy()
    eligible = observable_positive_mask(clean, corrupt, labels)
    np.testing.assert_array_equal(np.flatnonzero(eligible), np.arange(8, 22))
    cw, pw = phase_weights(labels, np.ones(32, bool), np.ones(32, bool), eligible)
    assert not pw[4:8].any()
    assert (cw[4:8] > 0).all()
    assert pw[8:16].sum() == pytest.approx(.25)
    assert pw[16:22].sum() == pytest.approx(.25)
    assert cw.sum() + pw[labels == 0].sum() == pytest.approx(.5)
    np.testing.assert_array_equal(labels, original_labels)


def test_positive_mask_restarts_for_each_event_and_ignores_unobserved_changes():
    clean = Record(np.zeros((32, 2), np.float32), np.ones((32, 2), bool))
    corrupt = Record(clean.z.copy(), clean.mask.copy())
    labels = np.zeros(32, np.int8)
    labels[3:9] = labels[14:24] = 1
    corrupt.z[5:9, 0] = 1
    corrupt.z[14:24, 1] = 2
    clean.mask[:, 1] = corrupt.mask[:, 1] = False
    corrupt.z[18:24, 0] = 1
    eligible = observable_positive_mask(clean, corrupt, labels)
    expected = np.zeros(32, bool)
    expected[5:9] = expected[18:24] = True
    np.testing.assert_array_equal(eligible, expected)
    _, pw = phase_weights(labels, np.ones(32, bool), np.ones(32, bool), eligible)
    assert pw[3:9].sum() == pytest.approx(.25)
    assert pw[14:24].sum() == pytest.approx(.25)
    corrupt.mask[:, 0] = False
    with pytest.raises(ValueError, match="No scorable positive"):
        phase_weights(labels, np.ones(32, bool), np.ones(32, bool),
                      observable_positive_mask(clean, corrupt, labels))


def test_task_cache_stores_corrected_weights_and_rejects_legacy_supervision(tmp_path, monkeypatch):
    from asei import training
    from asei.cache import TaskFeatureStore

    cfg = load_config("configs/default.yaml")
    cfg["stage2"].update(train_pairs=1, validation_pairs=1, families=["variance"])
    cfg["runtime"].update(task_cache_format="packed", log_interval_seconds=0)
    clean = Record(np.zeros((100, 2), np.float32), np.ones((100, 2), bool), 32)
    corrupt = Record(clean.z.copy(), clean.mask.copy(), 32)
    corrupt.z[48:60, 0] = 1
    labels = np.zeros(100, np.int8)
    labels[40:60] = 1

    def fake_pair(*args, **kwargs):
        return TaskPair(clean, corrupt, labels, {"events": [{"start": 40, "stop": 60}]})

    def fake_features(record, model, batch_size):
        n, c = record.z.shape
        return FeatureSequence(record.z, record.mask, np.zeros((n, c, 2), np.float32),
                               np.full((n, c), -1, np.int32), np.full(n, -1, np.int32), 32)

    monkeypatch.setattr(training, "generate_pair", fake_pair)
    monkeypatch.setattr(training, "extract_features", fake_features)
    model = SimpleNamespace(cfg={"window": 32})
    directory = tmp_path / "cache"
    key, _, _ = training.build_task_cache(model, clean, clean, cfg, directory, 42, "representation")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["supervision_protocol"] == TASK_SUPERVISION_PROTOCOL
    assert all(recipe["unidentifiable_positive_points"] == 8 for recipe in manifest["recipes"])
    store = TaskFeatureStore(directory, key)
    try:
        for split in ("train", "validation"):
            sequence = store.get(split, 0, "corrupt")
            assert sequence.labels[40:48].all()
            assert not sequence.weights[40:48].any()
            assert sequence.weights[sequence.labels == 1].sum() == pytest.approx(.5)
        del sequence
    finally:
        store.close()
    settings = {k: cfg["stage2"][k] for k in
                ("families", "train_pairs", "validation_pairs", "validation_seeds")}
    legacy = fingerprint({"version": 1, "representation": "representation", "seed": 42, "tasks": settings})
    assert key != legacy
    manifest["key"] = legacy
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Task cache configuration differs"):
        training.build_task_cache(model, clean, clean, cfg, directory, 42, "representation")


def test_diagnostics_on_completed_compact_run_reuses_models_and_creates_evidence(tmp_path, monkeypatch):
    from asei import pipeline, resources
    from asei.metrics import LABEL_METRICS, SCORE_METRICS

    torch.set_num_threads(1)
    monkeypatch.setattr(resources, "_governor", None)
    monkeypatch.setattr(resources, "_governor_settings", None)
    monkeypatch.setattr(resources, "_cpu_counters", lambda: None)
    monkeypatch.setattr(resources, "_set_affinity", lambda cpus: False)
    cfg = load_config("configs/default.yaml")
    cfg["model"].update(window=32, d_model=8, heads=2, layers=1, dropout=0.,
                         marginal_prototypes=4, joint_prototypes=4)
    cfg["stage1"].update(epochs=1, batch_size=32, stride=8, workers=0)
    cfg["stage2"].update(epochs=1, batch_pairs=2, block_length=32, train_pairs=4, validation_pairs=4)
    cfg["runtime"].update(output=str(tmp_path / "outputs"), device="cpu", threads=1,
                          feature_batch_size=64, keep_intermediates=False, log_interval_seconds=0)
    frame = pd.DataFrame(np.random.default_rng(17).normal(size=(580, 3)), columns=["a", "b", "c"])
    frame["label"] = 0
    frame.loc[460:463, "label"] = 1
    path = tmp_path / "PSM.csv"
    frame.to_csv(path, index=False)
    entity = Entity("PSM", "PSM", path, 400)
    directory = pipeline.run_entity(entity, cfg, 42)
    before = load_torch_checkpoint(directory / "bundle.pt", map_location="cpu")
    assert not (directory / "predictions.npz").exists()
    assert "event" not in before["experiment"]["result"]["metrics"]
    cfg["runtime"]["run_id"] = before["experiment"]["result"]["run_id"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic evaluation must reuse the selected networks")

    for name in ("fit_representation", "fit_detector", "build_task_cache"):
        monkeypatch.setattr(pipeline, name, forbidden)
    pipeline.run_entity(entity, cfg, 42, diagnostics=True)
    after = load_torch_checkpoint(directory / "bundle.pt", map_location="cpu")
    for group in ("representation", "detector"):
        for name in before[group]:
            assert torch.equal(before[group][name], after[group][name])
    assert before["threshold"] == after["threshold"]
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert result["diagnostics"] is True
    assert set(result["metrics"]["label"]) == LABEL_METRICS
    assert set(result["metrics"]["score"]) == SCORE_METRICS
    assert result["metrics"]["event"]["events"] == 1
    with np.load(directory / "predictions.npz") as data:
        assert data["attribution"].shape == data["scores"].shape == (180,)
        for name in ("retention", "state_difference", "marginal_difference", "joint_difference"):
            assert data[name].shape == (180, 3)
    assert cfg["runtime"]["keep_intermediates"] is False
