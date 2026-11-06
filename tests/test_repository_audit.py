import numpy as np
import pandas as pd
import pytest
import torch

from asei.data import Record, Standardizer, read_frame
from asei.data_process import process_data_df, is_st
from asei.features import extract_features, score_features, StreamingDetector
from asei.model import NormalRepresentation, DualTrajectoryDetector


def test_long_format_aligns_timestamps_and_preserves_named_label(tmp_path):
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"] * 3,
                          "data": [10, 20, 0, 1, 30, 40],
                          "cols": ["a", "a", "label", "label", "b", "b"]})
    frame.iloc[[4, 5]] = frame.iloc[[5, 4]].to_numpy()
    path = tmp_path / "PSM.csv"
    frame.to_csv(path, index=False)
    actual = read_frame(path)
    assert actual.columns.tolist() == ["a", "label", "b"]
    np.testing.assert_array_equal(actual.to_numpy(), [[10, 0, 30], [20, 1, 40]])
    with pytest.raises(ValueError, match="timestamp"):
        process_data_df(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="timestamp"):
        process_data_df(frame.iloc[:-1])


def test_date_free_long_format_supports_interleaved_channels():
    frame = pd.DataFrame({"data": [1, 0, 2, 1], "cols": ["a", "label", "a", "label"]})
    np.testing.assert_array_equal(process_data_df(frame).to_numpy(), [[1, 0], [2, 1]])
    assert not is_st(pd.DataFrame(columns=["date", "data", "cols", "unused"]))
    assert is_st(pd.DataFrame(columns=["date", "data", "cols", "id"]))


def small_model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = NormalRepresentation(dict(channels=3, window=32, d_model=8, heads=2,
        layers=1, dropout=0., marginal_prototypes=4, joint_prototypes=4)).eval()
    model.memory.cache_decoders(model.history.identity.weight)
    return model


def test_relation_prediction_excludes_target_history_and_projection_is_feasible():
    model = small_model()
    history = torch.randn(2, 32, 3)
    mask = torch.ones_like(history, dtype=torch.bool)
    changed = history.clone()
    changed[:, :, 0] += 5
    with torch.no_grad():
        before, after = model.predict(history, mask), model.predict(changed, mask)
        torch.testing.assert_close(before[:, 0, 1], after[:, 0, 1])
        model.relation.copy_(torch.randn_like(model.relation) * 5)
        model.project_relation()
    torch.testing.assert_close(model.relation.sum(-1), torch.ones(3))
    assert model.relation.diagonal().eq(0).all()
    assert model.relation.min() >= 0 and model.relation.max() <= .5


def test_offline_block_step_and_stream_agree_with_missing_observations():
    model = small_model()
    detector = DualTrajectoryDetector(model.history.identity.weight).eval()
    raw = np.random.default_rng(42).normal(size=(86, 3))
    raw[40:43, 0] = np.nan
    scaler = Standardizer(np.zeros(3), np.ones(3))
    sequence = extract_features(scaler.transform(raw), model, 9)
    with torch.no_grad():
        block = score_features(sequence, model, detector, 13)
        step = score_features(sequence, model, detector, 17, execution="step")
    stream = StreamingDetector(model, detector, scaler, 0.)
    outputs = stream.process(raw)
    np.testing.assert_array_equal(block["valid"], [o["valid"] for o in outputs])
    np.testing.assert_allclose(block["scores"], step["scores"], atol=2e-6, equal_nan=True)
    np.testing.assert_allclose(block["scores"], [o["score"] for o in outputs], atol=2e-6, equal_nan=True)
    altered = raw.copy()
    altered[72:] += 100
    changed = score_features(extract_features(scaler.transform(altered), model, 7), model, detector, 11)
    np.testing.assert_allclose(block["scores"][:72], changed["scores"][:72], atol=2e-6, equal_nan=True)


def test_nonfinite_ready_scores_fail_instead_of_being_removed_from_calibration():
    model = small_model()
    detector = DualTrajectoryDetector(model.history.identity.weight).eval()
    with torch.no_grad():
        detector.head[-1].bias.fill_(float("inf"))
    record = Record(np.zeros((40, 3), np.float32), np.ones((40, 3), bool))
    sequence = extract_features(record, model)
    for execution in ("step", "block"):
        with pytest.raises(FloatingPointError, match="Non-finite detector score"):
            score_features(sequence, model, detector, execution=execution)


def test_paired_statistics_reject_mismatched_metric_protocols(monkeypatch):
    from asei import aggregate
    frame = pd.DataFrame([dict(dataset="PSM", entity="PSM", seed=42, variant="full",
                               metrics_protocol="new", **{"score.auc_pr": .5})])
    other = frame.assign(metrics_protocol="old")
    monkeypatch.setattr(aggregate, "collect", lambda root, seeds: frame if root == "a" else other)
    with pytest.raises(ValueError, match="same metrics protocol"):
        aggregate.paired_comparison("a", "b", "score.auc_pr", "full", "full")
