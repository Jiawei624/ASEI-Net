import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from asei.cache import TaskFeatureStore
from asei.config import load_config
from asei.data import Record
from asei.features import FeatureSequence
from asei.tasks import generate_pair, phase_weights


def sparse_record(length=1800, channels=51, seed=19):
    values = np.zeros((length, channels), np.float32)
    values[:, -1] = np.random.default_rng(seed).normal(size=length)
    return Record(values, np.ones_like(values, bool), 32)


@pytest.mark.parametrize("index,seed,family", [(7, 1002, "relation"), (15, 20042, "relation"),
                                              (6, 20042, "temporal")])
def test_sparse_channels_produce_real_reproducible_events(index, seed, family):
    source, donor = sparse_record(), sparse_record(seed=23)
    before, donor_before = source.z.copy(), donor.z.copy()
    pair = generate_pair(source, donor, 32, index, seed)
    repeated = generate_pair(source, donor, 32, index, seed)
    assert pair.recipe == repeated.recipe
    assert pair.recipe["family"] == family
    assert pair.recipe["layout"] == 2
    if family == "relation":
        assert pair.recipe["fallback"] == "variable_channels"
        assert pair.recipe["attempt"] >= 80
    np.testing.assert_array_equal(pair.corrupt.z, repeated.corrupt.z)
    np.testing.assert_array_equal(pair.labels, repeated.labels)
    np.testing.assert_array_equal(source.z, before)
    np.testing.assert_array_equal(donor.z, donor_before)
    np.testing.assert_array_equal(pair.clean.mask, pair.corrupt.mask)
    np.testing.assert_array_equal(pair.clean.z[~pair.labels.astype(bool)],
                                  pair.corrupt.z[~pair.labels.astype(bool)])
    for event in pair.recipe["events"]:
        start, stop, channels = event["start"], event["stop"], event["channels"]
        assert 50 in channels
        if family == "relation":
            assert channels == [50]
        assert start >= 32 and np.all(pair.labels[start:stop] == 1)
        reference = pair.clean.z[start:stop, channels]
        replacement = pair.corrupt.z[start:stop, channels]
        assert np.any(np.abs(replacement - reference) > 1e-6)
        if family == "relation":
            assert 0 <= event["donor_start"] <= len(donor.z) - (stop - start)
            np.testing.assert_allclose(replacement.mean(0), reference.mean(0), atol=1e-6)
            np.testing.assert_allclose(replacement.std(0), reference.std(0), rtol=1e-5, atol=1e-6)
    ready = np.arange(len(pair.labels)) >= 32
    cw, pw = phase_weights(pair.labels, ready, ready)
    assert cw.sum() + pw[pair.labels == 0].sum() == pytest.approx(.5)
    assert pw[pair.labels == 1].sum() == pytest.approx(.5)


@pytest.mark.parametrize("family", ["relation", "temporal"])
def test_impossible_constant_tasks_fail_instead_of_creating_false_labels(family):
    record = Record(np.zeros((700, 4), np.float32), np.ones((700, 4), bool))
    with pytest.raises(ValueError, match=r"seed=42, index=1, attempts=160"):
        generate_pair(record, record, 32, 1, 42, [family])


def test_fallback_respects_missing_channels_and_causal_fill():
    source, donor = sparse_record(), sparse_record(seed=23)
    source.mask[:, 0] = False
    source.mask[::7, -1] = False
    donor.mask[::11, -1] = False
    # Leave complete spans long enough to inject the four-point relation events.
    last = np.zeros(source.z.shape[1], np.float32)
    for i in range(len(source.z)):
        last = np.where(source.mask[i], source.z[i], last)
        source.z[i] = last
    pair = generate_pair(source, donor, 32, 7, 1002)
    for event in pair.recipe["events"]:
        assert pair.clean.mask[event["start"]:event["stop"], event["channels"]].all()
        assert 0 not in event["channels"]
    for i in range(1, len(pair.corrupt.z)):
        missing = ~pair.corrupt.mask[i]
        np.testing.assert_array_equal(pair.corrupt.z[i, missing], pair.corrupt.z[i - 1, missing])


def test_partial_packed_cache_resumes_without_changing_committed_pairs(tmp_path, monkeypatch):
    from asei import training

    cfg = load_config("configs/default.yaml")
    cfg["stage2"].update(train_pairs=16, validation_pairs=8)
    cfg["runtime"].update(task_cache_format="packed", task_cache_checkpoint_pairs=16,
                          log_interval_seconds=0)
    model = SimpleNamespace(cfg={"window": 32})
    source = sparse_record(length=700)
    donor = sparse_record(length=800, seed=23)
    calls = []

    def fake_features(record, model, batch_size):
        n, c = record.z.shape
        return FeatureSequence(record.z, record.mask, np.zeros((n, c, 2), np.float32),
                               np.full((n, c), -1, np.int32), np.full(n, -1, np.int32), 32)

    def interrupted(source, donor, window, index, seed, families):
        if index == 15:
            raise ValueError("Interrupted relation task")
        return generate_pair(source, donor, window, index, seed, families)

    monkeypatch.setattr(training, "extract_features", fake_features)
    monkeypatch.setattr(training, "generate_pair", interrupted)
    directory = tmp_path / "task_cache"
    with pytest.raises(ValueError, match="Interrupted relation task"):
        training.build_task_cache(model, donor, source, cfg, directory, 42, "representation-key")
    before = json.loads((directory / "train/index.json").read_text())
    assert len(before["recipes"]) == 15
    assert not json.loads((directory / "manifest.json").read_text())["complete"]
    prefixes = {path: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in (directory / "train").glob("*.bin")}

    def record_calls(source, donor, window, index, seed, families):
        calls.append((index, seed))
        return generate_pair(source, donor, window, index, seed, families)

    monkeypatch.setattr(training, "generate_pair", record_calls)
    key, _, _ = training.build_task_cache(model, donor, source, cfg, directory, 42, "representation-key")
    after = json.loads((directory / "train/index.json").read_text())
    assert after["recipes"][:15] == before["recipes"]
    assert calls == [(15, 20042)] + [(i, [1001, 1002][i % 2]) for i in range(8)]
    for path, (length, digest) in prefixes.items():
        with path.open("rb") as handle:
            assert hashlib.sha256(handle.read(length)).hexdigest() == digest
    assert json.loads((directory / "manifest.json").read_text())["complete"]
    store = TaskFeatureStore(directory, key)
    try:
        for split, count in (("train", 16), ("validation", 8)):
            for i in range(count):
                pair = [store.get(split, i, role) for role in ("clean", "corrupt")]
                assert pair[1].labels.any()
                assert sum(item.weights.sum() for item in pair) == pytest.approx(1)
    finally:
        store.close()
