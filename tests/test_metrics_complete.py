import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
import torch

from asei.aggregate import aggregate
from asei.config import ROOT, load_config, write_json
from asei.data import Record, Standardizer, Entity
from asei.evaluation import METRICS
from asei.evaluation import classification_metrics_label as native_labels
from asei.evaluation import regression_metrics as native_regression
from asei.features import FeatureSequence
from asei.forecast_metrics import evaluate_forecasts
from asei.metrics import (LABEL_METRICS, SCORE_METRICS, REGRESSION_METRICS, evaluate,
                          add, label_metrics, score_metrics, regression_metrics, point_adjusted_predictions)


@pytest.fixture(autouse=True)
def small_cpu(monkeypatch):
    from asei import resources
    torch.set_num_threads(1)
    monkeypatch.setattr(resources, "_governor", None)
    monkeypatch.setattr(resources, "_governor_settings", None)
    monkeypatch.setattr(resources, "_cpu_counters", lambda: None)
    monkeypatch.setattr(resources, "_set_affinity", lambda cpus: False)


def test_every_documented_metric_is_registered():
    names = set(re.findall(r"^([A-Za-z_]+) \(", (ROOT / "metrics.txt").read_text(encoding="utf-8"), re.M))
    assert len(names) == 38
    assert names == LABEL_METRICS | SCORE_METRICS | REGRESSION_METRICS == set(METRICS)


@pytest.mark.parametrize("profile", ["default", "psm", "smd", "swat", "pump"])
def test_every_profile_enables_all_documented_metrics(profile):
    cfg = load_config("configs/" + profile + ".yaml")["evaluation"]
    assert set(cfg["label_metrics"]) == LABEL_METRICS
    assert set(cfg["score_metrics"]) == SCORE_METRICS
    assert cfg["regression_metrics"] == []


def test_point_adjustment_and_add_use_separate_alarm_sequences():
    actual = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
    alarms = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    before = alarms.copy()
    scores = np.where(alarms, .9, .1)
    cfg = load_config("configs/default.yaml")["evaluation"]
    result, returned = evaluate(actual, scores, np.ones(10, bool), .5, cfg)
    values = result["label"]
    assert set(values) == LABEL_METRICS
    assert values["f_score"] == pytest.approx(2 / 7)
    assert values["adjust_precision"] == .75 and values["adjust_recall"] == .6
    assert values["adjust_f_score"] == pytest.approx(2 / 3)
    assert values["adjust_accuracy"] == .7
    assert values["add"] == 2
    np.testing.assert_array_equal(point_adjusted_predictions(actual, alarms), [0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(alarms, before)
    np.testing.assert_array_equal(returned, before)


@pytest.mark.parametrize("truth,prediction,expected", [(0, 0, 0), (0, 1, 0), (1, 1, 1), (1, 0, 0)])
@pytest.mark.parametrize("name", ["precision", "recall", "f_score", "adjust_precision", "adjust_recall", "adjust_f_score"])
def test_positive_class_metrics_handle_single_class_inputs(name, truth, prediction, expected):
    actual, predicted = np.full(8, truth), np.full(8, prediction)
    assert getattr(native_labels, name)(actual, predicted) == expected
    values = label_metrics(actual, predicted, [name], {})
    assert values[name] == expected


def test_no_anomaly_metrics_keep_keys_and_explain_undefined_values(tmp_path):
    actual = np.zeros(8, int)
    cfg = load_config("configs/default.yaml")["evaluation"]
    result, _ = evaluate(actual, np.zeros(8), np.ones(8, bool), 1, cfg)
    assert result["label"]["accuracy"] == result["label"]["adjust_accuracy"] == 1
    assert set(result["score"]) == SCORE_METRICS and np.isnan(result["label"]["add"])
    assert "add" in result["undefined_or_warnings"]
    path = tmp_path / "result.json"
    write_json(path, result)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    assert decoded["label"]["add"] is None


def test_single_all_anomaly_range_metrics_are_defined():
    values = label_metrics(np.ones(8, int), np.ones(8, int), ["rprecision", "rrecall", "rf"], {})
    assert values == {"rprecision": 1, "rrecall": 1, "rf": 1}


def test_score_metrics_match_fixed_reference_values():
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([.1, .2, .9, .4, .1, .3, .2, .6, .7, .8, .1, .2, .1, .4, .1, .1])
    result = score_metrics(y, scores, sorted(SCORE_METRICS), {})
    assert result == pytest.approx({"auc_roc": .9909090909090909, "auc_pr": .9666666666666667,
        "R_AUC_ROC": .9578692121225052, "R_AUC_PR": .9454845870221293,
        "VUS_ROC": .9642539492713234, "VUS_PR": .9549847182737853})


class FixedScaler:
    def transform(self, values):
        return (np.asarray(values, float) - np.array([1., 2.])) / np.array([2., 4.])


def test_all_sixteen_regression_metrics_and_mase_seasonality_two():
    actual = np.array([[2, 100], [4, 100]])
    predicted = np.array([[3, 102], [3, 98]])
    history = np.array([[1, 10], [2, 12], [3, 14], [4, 16]])
    result, errors = regression_metrics(actual, predicted, sorted(REGRESSION_METRICS),
                                        scaler=FixedScaler(), hist_data=history, seasonality=1)
    assert not errors and set(result) == REGRESSION_METRICS
    assert result["mae"] == 1.5 and result["mse"] == 2.5
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["mape"] == pytest.approx(19.75)
    assert result["wape"] == pytest.approx(600 / 206)
    assert result["mae_norm"] == .5 and result["mse_norm"] == .25 and result["rmse_norm"] == .5
    assert result["mape_norm"] == pytest.approx(5050 / 147)
    assert result["smape_norm"] == pytest.approx(25 * (2 / 3 + 2 / 99 + 2 / 5 + 2 / 97))
    assert result["wape_norm"] == pytest.approx(200 / 51)
    assert result["msmape_norm"] == pytest.approx(25 * (5 / 8 + 5 / 248 + 5 / 13 + 5 / 243))
    assert result["mase"] == result["mase_norm"] == 1
    assert native_regression.mase(actual, predicted, history, seasonality=2) == .5
    expected_smape = np.mean(2 * np.abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100
    expected_msmape = np.mean(2 * np.abs(actual - predicted) / np.maximum(.6, abs(actual) + abs(predicted) + .1)) * 100
    assert result["smape"] == pytest.approx(expected_smape)
    assert result["msmape"] == pytest.approx(expected_msmape)


def test_mase_preserves_missing_history_gaps_and_marks_constant_scale_undefined():
    assert native_regression.mase(np.array([10]), np.array([8]), np.array([0, np.nan, 4, 6, 8])) == 1
    result, errors = regression_metrics([1, 2], [2, 3], ["mase"], hist_data=np.ones(8))
    assert np.isnan(result["mase"]) and "mase" in errors
    assert native_regression.msmape(np.array([0]), np.array([.2])) == pytest.approx(200 / 3)


def test_branch_forecast_pooling_uses_observed_targets_and_training_only_history():
    actual = np.array([[99, 99], [2, 100], [4, 100]], np.float32)
    predicted = np.array([[99, 99], [3, 102], [3, 98]], np.float32)
    scale = np.array([2., 4.])
    mean = np.array([1., 2.])
    training_raw = np.array([[1, 10], [2, 12], [3, 14], [4, 16]])
    training = Record(((training_raw - mean) / scale).astype(np.float32), np.ones((4, 2), bool))
    sequence = FeatureSequence(((actual - mean) / scale).astype(np.float32), np.ones((3, 2), bool),
        np.stack([(predicted - mean) / scale, (predicted - mean) / scale], -1).astype(np.float32),
        np.full((3, 2), -1), np.full(3, -1), 1)
    values, details = evaluate_forecasts(sequence, training, Standardizer(mean, scale),
                                          sorted(REGRESSION_METRICS))
    for branch in ("self", "relation"):
        assert set(values[branch]) == REGRESSION_METRICS
        assert values[branch]["rmse"] == pytest.approx(np.sqrt(2.5))
        assert values[branch]["wape"] == pytest.approx(600 / 206)
        assert values[branch]["wape_norm"] == pytest.approx(200 / 51)
        assert values[branch]["mase"] == values[branch]["mase_norm"] == 1
        assert details["coverage"][branch]["mae"]["targets"] == 4
    sequence.mask[-1, 1] = False
    masked, details = evaluate_forecasts(sequence, training, Standardizer(mean, scale), ["mse"])
    assert masked["self"]["mse"] == 2
    assert details["coverage"]["self"]["mse"]["targets"] == 3


def test_undefined_metrics_are_retained_as_empty_summary_columns(tmp_path):
    root = tmp_path / "outputs" / "one"
    root.mkdir(parents=True)
    write_json(root / "result.json", {"dataset": "PSM", "entity": "PSM", "seed": 42, "variant": "full",
        "metrics": {"regression": {"self": {"mape": None, "mae": 1.5}}, "label": {"adjust_f_score": .8}}})
    output = aggregate(tmp_path / "outputs", tmp_path / "summary", seeds=[42])
    frame = pd.read_csv(output / "dataset_metrics.csv")
    assert "self.mape" not in frame and "add" in frame and frame["add"].isna().all()
    assert frame["adjust_f_score"].iloc[0] == .8


def test_two_stage_run_saves_all_metrics_and_reuses_weights_for_evaluation(tmp_path, monkeypatch):
    from asei.pipeline import run_entity
    cfg = load_config("configs/default.yaml")
    cfg["model"].update(window=32, d_model=16, layers=1, dropout=0., marginal_prototypes=4, joint_prototypes=8)
    cfg["stage1"].update(epochs=1, batch_size=32, stride=8, workers=0)
    cfg["stage2"].update(epochs=1, batch_pairs=2, block_length=32, train_pairs=4, validation_pairs=4)
    cfg["evaluation"]["latency_points"] = 1
    cfg["runtime"].update(output=str(tmp_path / "outputs"), device="cpu", threads=1, feature_batch_size=64,
                          keep_intermediates=True)
    values = np.random.default_rng(17).normal(size=(580, 3))
    frame = pd.DataFrame(values, columns=["a", "b", "c"])
    frame["label"] = 0
    frame.loc[460:463, "label"] = 1
    path = tmp_path / "PSM.csv"
    frame.to_csv(path, index=False)
    entity = Entity("PSM", "PSM", path, 400)
    out = run_entity(entity, cfg, 42)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert set(result["metrics"]["label"]) == LABEL_METRICS
    assert set(result["metrics"]["score"]) == SCORE_METRICS
    assert "regression" not in result["metrics"] and "event" not in result["metrics"]
    with np.load(out / "predictions.npz") as predictions:
        np.testing.assert_array_equal(predictions["point_adjusted_alarms"],
                                      point_adjusted_predictions(predictions["labels"], predictions["alarms"]))
        before = predictions["scores"].copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("Metric-only evaluation must not train")
    monkeypatch.setattr("asei.pipeline.fit_representation", forbidden)
    monkeypatch.setattr("asei.pipeline.fit_detector", forbidden)
    monkeypatch.setattr("asei.pipeline.build_task_cache", forbidden)
    cfg["runtime"]["run_id"] = result["run_id"]
    run_entity(entity, cfg, 42, stage="evaluate")
    with np.load(out / "predictions.npz") as predictions:
        np.testing.assert_array_equal(before, predictions["scores"])


def test_python38_source_grammar():
    for folder in ("asei", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=8)
