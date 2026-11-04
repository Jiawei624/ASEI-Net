import numpy as np
import pandas as pd
import pytest

from asei.data import discover, Entity, load_entity, Standardizer
from asei.metrics import add, event_metrics, threshold_from_normal, label_metrics, score_metrics, LABEL_METRICS, SCORE_METRICS
from asei.tab import tab_module


def test_tab_long_format_and_official_boundary(tmp_path):
    (tmp_path / "data").mkdir()
    values = {"a": np.arange(120, dtype=float), "b": np.arange(120) * 2.0,
              "label": np.r_[np.zeros(100), np.ones(20)]}
    frame = pd.concat([pd.DataFrame({"data": v, "cols": name}) for name, v in values.items()], ignore_index=True)
    path = tmp_path / "data" / "PSM_fixture.csv"
    frame.to_csv(path, index=False)
    pd.DataFrame([{"file_name": path.name, "train_lens": 100}]).to_csv(tmp_path / "DETECT_META.csv", index=False)
    entity, = discover(tmp_path)
    train, valid, test, y, scaler, provenance = load_entity(entity, 0.8)
    assert (len(train.z), len(valid.z), len(test.z)) == (80, 20, 20)
    np.testing.assert_allclose(scaler.mean, [39.5, 79.0])
    assert y.sum() == 20 and provenance["official_train_lens"] == 100
    frame.loc[frame.cols == "label", "data"] = np.r_[np.ones(1), np.zeros(119)]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="contains anomalies"):
        load_entity(entity, 0.8)


def test_causal_fill_and_constant_channel():
    raw = np.array([[np.nan, 5.0], [2.0, 5.0], [np.nan, 5.0], [4.0, 5.0]])
    scaler = Standardizer.fit(raw)
    r = scaler.transform(raw)
    np.testing.assert_allclose(r.z[:, 0], [0, -1, -1, 1])
    assert scaler.scale[1] == 1
    assert not r.mask[0, 0] and not r.mask[2, 0]


def test_add_includes_missed_events_without_adjustment():
    y = np.array([1, 1, 1, 0, 1, 1, 0, 1])
    p = np.array([0, 0, 1, 0, 0, 0, 0, 1])
    assert add(y, p) == pytest.approx((2 + 2 + 0) / 3)
    assert np.isnan(add(np.zeros(3), np.zeros(3)))
    np.testing.assert_array_equal(p, [0, 0, 1, 0, 0, 0, 0, 1])
    # The 8-point confirmation window contributes no extra delay.
    y = np.r_[np.zeros(2), np.ones(3), np.zeros(12)]
    p = np.r_[np.zeros(2), np.ones(5), np.zeros(10)]
    event = event_metrics(y, p)
    assert event["recovery_delay_observed_mean"] == 2
    assert event["recovery_censored_fraction"] == 0


def test_strict_order_statistic_threshold():
    point = threshold_from_normal(np.arange(100), 0.01)
    assert point["threshold"] == 98 and point["normal_exceedance"] == 0.01
    assert threshold_from_normal([1, 1, 1], 0.01)["normal_exceedance"] == 0


def test_metric_dispatch_matches_pinned_tab():
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.3, 0.2, 0.6, 0.7, 0.8, 0.1, 0.2, 0.1, 0.4, 0.1, 0.1])
    p = (s > 0.5).astype(int)
    errors = {}
    labels = label_metrics(y, p, sorted(LABEL_METRICS), errors)
    scores = score_metrics(y, s, sorted(SCORE_METRICS), errors)
    lm = tab_module("evaluation.metrics.classification_metrics_label")
    sm = tab_module("evaluation.metrics.classification_metrics_score")
    for name, value in labels.items():
        if name != "add":
            assert value == pytest.approx(getattr(lm, name)(y.copy(), p.copy()), nan_ok=True)
    for name, value in scores.items():
        assert value == pytest.approx(getattr(sm, name)(y.copy(), s.copy()), nan_ok=True)
    with pytest.raises(ValueError, match="threshold scans"):
        score_metrics(y, s, ["best_f_score"], {})
