import json

import numpy as np
import pandas as pd
import pytest

from asei.data import Entity
from asei.pipeline import run_entity, load_bundle
from asei.aggregate import aggregate


@pytest.mark.parametrize("continuous,independent_validation", [(False, False), (True, False), (True, True)])
def test_two_stage_checkpoint_calibration_evaluation_and_bundle(tmp_path, tiny_config, monkeypatch, continuous, independent_validation):
    rng = np.random.default_rng(31)
    t = np.arange(580)
    base = np.stack([np.sin(t / 9), np.cos(t / 13), np.sin(t / 9) * 0.5 + np.cos(t / 13)], 1)
    x = base + rng.normal(0, 0.05, base.shape)
    y = np.zeros(len(t), int)
    x[460:490, 0] += 1
    y[460:490] = 1
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    frame["label"] = y
    path = tmp_path / "PSM_fixture.csv"
    frame.to_csv(path, index=False)
    validation_path = None
    if independent_validation:
        validation_path = tmp_path / "normal_validation.csv"
        frame.iloc[100:180].to_csv(validation_path, index=False)
    entity = Entity("PSM", "fixture", path, 400, continuous, validation_path)
    tiny_config["runtime"]["output"] = str(tmp_path / "outputs")
    out = run_entity(entity, tiny_config, 0, diagnostics=True)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["test_points"] == 180
    assert result["calibration"]["normal_scores"] == (48 if independent_validation else 80)
    assert result["metrics"]["point_adjustment_in_primary_metrics"] is False
    with np.load(out / "predictions.npz") as data:
        assert data["scores"].shape == data["labels"].shape == (180,)
        assert np.isfinite(data["scores"][:32]).all() if continuous else np.isnan(data["scores"][:32]).all()
        before = data["scores"].copy()
    stream, names = load_bundle(out / "bundle.pt")
    assert names == ["a", "b", "c"]
    if continuous:
        stream.process(x[:400])
    output = stream.process(x[400:])
    np.testing.assert_allclose(before, [p["score"] for p in output], equal_nan=True, atol=3e-6, rtol=3e-5)
    # Evaluation-only reuse must not train or regenerate tasks.
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation attempted training.")
    monkeypatch.setattr("asei.pipeline.fit_representation", forbidden)
    monkeypatch.setattr("asei.pipeline.fit_detector", forbidden)
    monkeypatch.setattr("asei.pipeline.build_task_cache", forbidden)
    run_entity(entity, tiny_config, 0, stage="evaluate")
    with np.load(out / "predictions.npz") as data:
        np.testing.assert_array_equal(before, data["scores"])
    summary = aggregate(tmp_path / "outputs", tmp_path / "summary")
    assert (summary / "dataset_summary.csv").is_file()
