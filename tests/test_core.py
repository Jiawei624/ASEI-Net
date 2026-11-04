import copy

import numpy as np
import pytest
import torch

from asei.data import Record, Standardizer
from asei.features import extract_features, score_features, StreamingDetector
from asei.model import NormalRepresentation, DualTrajectoryDetector
from asei.training import initialize_prototypes, finalize_memory, revive_unused
from asei.tasks import event_ranges, generate_pair, phase_weights


def normal_record(n=200, channels=3):
    rng = np.random.default_rng(14)
    z = rng.normal(size=(n, channels)).astype(np.float32)
    return Record(z, np.ones_like(z, bool))


def initialized_model(cfg, record):
    model = NormalRepresentation(cfg["model"])
    model.descriptor.fit(record)
    initialize_prototypes(model, record)
    finalize_memory(model, record)
    return model.eval()


def test_prediction_isolation_and_relation_constraints(tiny_config):
    model = NormalRepresentation(tiny_config["model"]).eval()
    history, mask = torch.randn(2, 32, 3), torch.ones(2, 32, 3, dtype=torch.bool)
    with torch.no_grad():
        before = model.predict(history, mask)
        history[:, :, 0] += torch.linspace(1, 8, 32)
        after = model.predict(history, mask)
    torch.testing.assert_close(before[:, 0, 1], after[:, 0, 1], atol=1e-7, rtol=1e-7)
    assert not torch.allclose(before[:, 0, 0], after[:, 0, 0])
    for c in (1, 2, 3, 7):
        cfg = dict(tiny_config["model"], channels=c)
        m = NormalRepresentation(cfg)
        with torch.no_grad():
            m.relation.normal_()
        m.project_relation()
        a = m.relation.detach()
        assert torch.diagonal(a).eq(0).all()
        if c > 1:
            torch.testing.assert_close(a.sum(1), torch.ones(c), atol=2e-6, rtol=0)
            assert (a >= 0).all() and (a <= max(0.5, 1 / (c - 1)) + 1e-6).all()


def test_shared_trajectory_equality_and_missing_reset(tiny_config):
    record = normal_record()
    model = initialized_model(tiny_config, record)
    features = extract_features(record, model)
    f = {k: v[32:33] for k, v in features.block(0, 34, model).items()}
    f["chi"] = f["joint"].clone()
    detector = DualTrajectoryDetector(model.history.identity.weight)
    state = None
    for _ in range(4):
        _, state = detector.step(f, state)
        torch.testing.assert_close(state.observed, state.normal, atol=1e-7, rtol=0)
    f["m_valid"] = torch.zeros_like(f["m_valid"])
    f["j_valid"] = torch.zeros_like(f["j_valid"])
    out, state = detector.step(f, state)
    assert out["valid"].item()  # Prediction evidence remains available.
    assert state.observed.eq(0).all() and state.normal.eq(0).all()
    f["observed"] = torch.zeros_like(f["observed"])
    out, _ = detector.step(f, state)
    assert not out["valid"].item()


def test_offline_streaming_and_chunk_equivalence(tiny_config):
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(100, 3))
    raw[0, 1] = np.nan
    raw[50:52, 2] = np.nan
    raw[80] = np.nan
    scaler = Standardizer.fit(raw[:60])
    record = scaler.transform(raw)
    model = initialized_model(tiny_config, record)
    detector = DualTrajectoryDetector(model.history.identity.weight).eval()
    sequence = extract_features(record, model, 17)
    a = score_features(sequence, model, detector, 13)
    b = score_features(sequence, model, detector, 64)
    np.testing.assert_allclose(a["scores"], b["scores"], equal_nan=True, atol=1e-6)
    stream = StreamingDetector(model, detector, scaler, 0)
    output = stream.process(raw[:47]) + stream.process(raw[47:])
    np.testing.assert_allclose(a["scores"], [o["score"] for o in output], equal_nan=True, atol=3e-6, rtol=3e-5)
    np.testing.assert_array_equal(a["valid"], [o["valid"] for o in output])
    # A future edit must not change any earlier score.
    changed = raw.copy()
    changed[75:] += 5
    future = extract_features(scaler.transform(changed), model)
    c = score_features(future, model, detector)
    np.testing.assert_allclose(a["scores"][:75], c["scores"][:75], equal_nan=True, atol=1e-6)


def test_task_families_balance_and_no_cross_boundary():
    source = normal_record(500)
    source.target_start = 100
    seen = set()
    for i in range(8):
        pair = generate_pair(source, source, 32, i, 100)
        seen.add((pair.recipe["family"], pair.recipe["layout"]))
        assert pair.recipe["source_start"] + 32 >= 100
        first = event_ranges(pair.labels)[0][0]
        np.testing.assert_array_equal(pair.clean.z[:first], pair.corrupt.z[:first])
        np.testing.assert_array_equal(pair.clean.z[~pair.labels.astype(bool)], pair.corrupt.z[~pair.labels.astype(bool)])
        ready = np.arange(len(pair.labels)) >= 32
        cw, pw = phase_weights(pair.labels, ready, ready)
        np.testing.assert_allclose(pw[pair.labels == 1].sum(), 0.5, atol=1e-6)
        np.testing.assert_allclose(cw.sum() + pw[pair.labels == 0].sum(), 0.5, atol=1e-6)
        for event in pair.recipe["events"]:
            if pair.recipe["family"] == "relation":
                s, e, ch = event["start"], event["stop"], event["channels"]
                np.testing.assert_allclose(pair.clean.z[s:e, ch].mean(0), pair.corrupt.z[s:e, ch].mean(0), atol=1e-6)
                np.testing.assert_allclose(pair.clean.z[s:e, ch].std(0), pair.corrupt.z[s:e, ch].std(0), atol=1e-6)
    assert len(seen) == 8


def test_single_channel_and_prediction_only(tiny_config):
    cfg = copy.deepcopy(tiny_config)
    cfg["model"]["channels"] = 1
    record = normal_record(channels=1)
    model = initialized_model(cfg, record)
    detector = DualTrajectoryDetector(model.history.identity.weight, sources=[])
    seq = extract_features(record, model)
    output = score_features(seq, model, detector)
    assert np.isfinite(output["scores"][32:]).all()
    np.testing.assert_array_equal(seq.prediction[:, :, 1], 0)


def test_dead_prototype_revival_clears_optimizer_momentum(tiny_config):
    record = normal_record(100)
    model = NormalRepresentation(tiny_config["model"])
    model.descriptor.fit(record)
    initialize_prototypes(model, record)
    memory = model.memory
    optimizer = torch.optim.Adam([memory.m_codes, memory.j_codes])
    for p in (memory.m_codes, memory.j_codes):
        p.grad = torch.ones_like(p)
    optimizer.step()
    memory.m_unused.fill_(4)
    memory.j_unused.fill_(4)
    revive_unused(model, optimizer, record, torch.zeros_like(memory.m_unused),
                   torch.zeros_like(memory.j_unused), np.random.default_rng(9))
    assert memory.m_unused.eq(0).all() and memory.j_unused.eq(0).all()
    for p in (memory.m_codes, memory.j_codes):
        assert optimizer.state[p]["exp_avg"].eq(0).all()
        assert optimizer.state[p]["exp_avg_sq"].eq(0).all()
        assert torch.isfinite(p).all()


def test_perturbed_missing_values_follow_own_past():
    source = normal_record(300)
    source.mask[4::5] = False
    # Construct the clean record with the same causal filling used by the loader.
    for t in range(1, len(source.z)):
        source.z[t] = np.where(source.mask[t], source.z[t], source.z[t - 1])
    propagated = False
    for seed in range(12):
        pair = generate_pair(source, source, 32, 0, seed)
        absent = ~pair.corrupt.mask[1:]
        np.testing.assert_array_equal(pair.corrupt.z[1:][absent], pair.corrupt.z[:-1][absent])
        propagated |= bool(np.any(pair.corrupt.z[~pair.corrupt.mask] != pair.clean.z[~pair.clean.mask]))
    assert propagated


@pytest.mark.parametrize("variant", ["mlp", "single_gru", "no_alignment", "zero_alignment", "fixed_gate", "independent_gru"])
def test_ablation_gradient_paths(tiny_config, variant):
    model = initialized_model(tiny_config, normal_record(70))
    features = extract_features(normal_record(70), model)
    detector = DualTrajectoryDetector(model.history.identity.weight, variant=variant)
    f = {k: v[0:1] for k, v in features.block(32, 33, model).items()}
    out, _ = detector.step(f)
    out["score"].sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in detector.head.parameters())
