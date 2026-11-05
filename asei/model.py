from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import Record
from .compat import transformer_encoder
from .resources import cooperate


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


def segment_arrays(record: Record, indices: np.ndarray):
    offsets = np.arange(-31, 1)
    return record.z[indices[:, None] + offsets], record.mask[indices[:, None] + offsets]


class Descriptor(nn.Module):
    """Five amplitude coordinates and a 32-sample locally standardized shape."""

    def __init__(self, channels):
        super().__init__()
        self.register_buffer("amp_mean", torch.zeros(channels, 5, dtype=torch.float64))
        self.register_buffer("amp_scale", torch.ones(channels, 5, dtype=torch.float64))

    @staticmethod
    def raw(segment):
        # Input [B,32,C], output [B,C,5], [B,C,32].
        segment = segment.transpose(1, 2)
        mean8, mean32 = segment[..., -8:].mean(-1), segment.mean(-1)
        var8 = segment[..., -8:].var(-1, unbiased=False)
        var32 = segment.var(-1, unbiased=False)
        amp = torch.stack([segment[..., -1], mean8, mean32,
                           (var8 + 1e-4).log(), (var32 + 1e-4).log()], -1)
        shape = (segment - mean32[..., None]) / (var32[..., None] + 1e-4).sqrt()
        return amp, shape

    @torch.no_grad()
    def fit(self, record: Record, batch=2048):
        c = record.z.shape[1]
        sums, squares = np.zeros((c, 5)), np.zeros((c, 5))
        counts = np.zeros(c, dtype=np.int64)
        for left in range(31, len(record.z), batch):
            cooperate()
            seg, masks = segment_arrays(record, np.arange(left, min(left + batch, len(record.z))))
            amp, _ = self.raw(torch.from_numpy(seg).double())
            valid = masks.all(1)
            a = amp.numpy()
            sums += np.where(valid[..., None], a, 0).sum(0)
            squares += np.where(valid[..., None], a * a, 0).sum(0)
            counts += valid.sum(0)
        mean = sums / np.maximum(counts[:, None], 1)
        scale = np.sqrt(np.maximum(0, squares / np.maximum(counts[:, None], 1) - mean * mean))
        scale = np.maximum(scale, 1e-3)
        scale[counts == 0] = 1
        self.amp_mean.copy_(torch.from_numpy(mean))
        self.amp_scale.copy_(torch.from_numpy(scale))
        return counts

    def forward(self, segment):
        amp, shape = self.raw(segment.float())
        return torch.cat([(amp - self.amp_mean.float()) / self.amp_scale.float(), shape], -1)


class HistoryEncoder(nn.Module):
    def __init__(self, channels, window, d_model, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.window = window
        self.identity = nn.Embedding(channels, d_model)
        self.projections = nn.ModuleList([nn.Linear(2 * p, d_model) for p in (1, 8, 32)])
        self.scale = nn.Embedding(3, d_model)
        self.position = nn.Embedding(window, d_model)
        endpoints = torch.cat([torch.arange(window - 32, window),
                               torch.arange(7, window, 8), torch.arange(31, window, 32)])
        self.register_buffer("endpoints", endpoints)
        self.register_buffer("causal_mask", endpoints[None, :] > endpoints[:, None])
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = transformer_encoder(layer, layers, nn.LayerNorm(d_model))

    def forward(self, history, observed):
        b, w, c = history.shape
        if w != self.window:
            raise ValueError("Prediction history has an incorrect window length.")
        values, masks = history.transpose(1, 2), observed.transpose(1, 2).to(history.dtype)
        tokens = []
        for s, p in enumerate((1, 8, 32)):
            v = values[..., -32:, None] if p == 1 else values.unfold(-1, p, p)
            m = masks[..., -32:, None] if p == 1 else masks.unfold(-1, p, p)
            tokens.append(self.projections[s](torch.cat([v, m], -1)) + self.scale.weight[s])
        x = torch.cat(tokens, dim=2)
        x = x + self.position(self.endpoints)[None, None] + self.identity.weight[None, :, None]
        # Each sequence contains exactly one channel; no cross-channel attention.
        x = self.transformer(x.reshape(b * c, x.shape[2], -1), mask=self.causal_mask)
        return x[:, 31].reshape(b, c, -1)


class PrototypeMemory(nn.Module):
    def __init__(self, channels, identity_dim, km, kj):
        super().__init__()
        self.channels, self.km, self.kj = channels, km, kj
        self.m_encoder = mlp(37 + identity_dim, 64, 32)
        self.m_decoder = mlp(32 + identity_dim, 64, 37)
        self.j_encoder = mlp(32 * channels, 128, 64)
        self.j_decoder = mlp(64, 128, 37 * channels)
        self.m_codes = nn.Parameter(torch.randn(channels, km, 32) * 0.02)
        self.j_codes = nn.Parameter(torch.randn(kj, 64) * 0.02)
        self.register_buffer("m_active", torch.ones(channels, km, dtype=torch.bool))
        self.register_buffer("j_active", torch.ones(kj, dtype=torch.bool))
        self.register_buffer("m_cache", torch.zeros(channels, km, 37))
        self.register_buffer("j_cache", torch.zeros(kj, channels, 37))
        self.register_buffer("m_unused", torch.zeros(channels, km, dtype=torch.long))
        self.register_buffer("j_unused", torch.zeros(kj, dtype=torch.long))

    def encode(self, chi, identity):
        e = identity[None].expand(chi.shape[0], -1, -1)
        qm = self.m_encoder(torch.cat([chi, e], -1))
        qj = self.j_encoder(qm.flatten(1))
        return qm, qj

    def nearest(self, qm, qj):
        md = (qm.square().sum(-1, keepdim=True) + self.m_codes.square().sum(-1)[None]
              - 2 * torch.einsum("bcd,ckd->bck", qm, self.m_codes))
        jd = qj.square().sum(-1, keepdim=True) + self.j_codes.square().sum(-1)[None] - 2 * qj @ self.j_codes.T
        mi = md.masked_fill(~self.m_active[None], torch.inf).argmin(-1)
        ji = jd.masked_fill(~self.j_active[None], torch.inf).argmin(-1)
        return mi, ji

    @staticmethod
    def reconstruction_loss(reconstructed, target):
        return 0.5 * ((reconstructed[..., :5] - target[..., :5]).square().mean(-1)
                      + (reconstructed[..., 5:] - target[..., 5:]).square().mean(-1))

    def loss(self, chi, complete, identity):
        qm, qj = self.encode(chi, identity)
        mi, ji = self.nearest(qm, qj)
        pm = self.m_codes[torch.arange(self.channels, device=chi.device)[None], mi]
        pj = self.j_codes[ji]
        sm, sj = qm + (pm - qm).detach(), qj + (pj - qj).detach()
        rm = self.m_decoder(torch.cat([sm, identity[None].expand(chi.shape[0], -1, -1)], -1))
        rj = self.j_decoder(sj).reshape_as(chi)
        mloss = (self.reconstruction_loss(rm, chi) + (qm.detach() - pm).square().mean(-1)
                 + 0.25 * (qm - pm.detach()).square().mean(-1))
        jloss = (self.reconstruction_loss(rj, chi).mean(-1) + (qj.detach() - pj).square().mean(-1)
                 + 0.25 * (qj - pj.detach()).square().mean(-1))
        mv = complete & self.m_active.any(-1)[None]
        jv = complete.all(-1) & self.j_active.any()
        loss = (mloss * mv).sum() / mv.sum().clamp_min(1)
        loss = loss + (jloss * jv).sum() / jv.sum().clamp_min(1)
        return loss, {"qm": qm.detach(), "qj": qj.detach(), "mi": mi.detach(), "ji": ji.detach(),
                      "mv": mv, "jv": jv}

    @torch.no_grad()
    def cache_decoders(self, identity):
        e = identity[:, None].expand(-1, self.km, -1)
        self.m_cache.copy_(self.m_decoder(torch.cat([self.m_codes, e], -1)))
        self.j_cache.copy_(self.j_decoder(self.j_codes).reshape_as(self.j_cache))

    @torch.no_grad()
    def query(self, chi, complete, identity):
        qm, qj = self.encode(chi, identity)
        mi, ji = self.nearest(qm, qj)
        mi = mi.masked_fill(~(complete & self.m_active.any(-1)[None]), -1)
        ji = ji.masked_fill(~(complete.all(-1) & self.j_active.any()), -1)
        return mi, ji

    def references(self, mi, ji):
        mv, jv = mi >= 0, ji >= 0
        channel = torch.arange(self.channels, device=mi.device)
        rm = self.m_cache[channel, mi.clamp_min(0)] * mv[..., None]
        rj = self.j_cache[ji.clamp_min(0)] * jv[..., None, None]
        return rm, rj, mv, jv[..., None].expand_as(mv)


class NormalRepresentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c, d = cfg["channels"], cfg["d_model"]
        self.cfg = cfg
        self.descriptor = Descriptor(c)
        self.history = HistoryEncoder(c, cfg["window"], d, cfg["heads"], cfg["layers"], cfg["dropout"])
        self.self_head = mlp(d, d, 1)
        self.rel_head = mlp(2 * d, d, 1)
        a = (torch.ones(c, c) - torch.eye(c)) / max(c - 1, 1)
        self.relation = nn.Parameter(a, requires_grad=c > 1)
        self.register_buffer("off_diagonal", ~torch.eye(c, dtype=torch.bool))
        self.memory = PrototypeMemory(c, d, cfg["marginal_prototypes"], cfg["joint_prototypes"])

    def predict(self, history, masks):
        h = self.history(history, masks)
        own = self.self_head(h).squeeze(-1)
        if self.cfg["channels"] == 1:
            return torch.stack([own, torch.zeros_like(own)], -1)
        external = torch.einsum("ij,bjd->bid", self.relation * self.off_diagonal, h)
        identity = self.history.identity.weight[None].expand(h.shape[0], -1, -1)
        rel = self.rel_head(torch.cat([external, identity], -1)).squeeze(-1)
        return torch.stack([own, rel], -1)

    @torch.no_grad()
    def project_relation(self):
        c = self.cfg["channels"]
        if c == 1:
            self.relation.zero_()
            return
        cap = max(0.5, 1 / (c - 1))
        b = self.relation
        low = b.masked_fill(~self.off_diagonal, torch.inf).amin(1) - cap
        high = b.masked_fill(~self.off_diagonal, -torch.inf).amax(1)
        for _ in range(50):
            mid = (low + high) / 2
            total = ((b - mid[:, None]).clamp(0, cap) * self.off_diagonal).sum(1)
            low = torch.where(total > 1, mid, low)
            high = torch.where(total > 1, high, mid)
        b.copy_((b - ((low + high) / 2)[:, None]).clamp(0, cap) * self.off_diagonal)

    def normal_loss(self, batch):
        predictions = self.predict(batch["history"], batch["history_mask"])
        branches = 2 if self.cfg["channels"] > 1 else 1
        errors = F.huber_loss(predictions[..., :branches],
                              batch["target"][..., None].expand_as(predictions[..., :branches]), reduction="none")
        observed = batch["target_mask"]
        forecast_loss = (errors.mean(-1) * observed).sum() / observed.sum().clamp_min(1)
        chi = self.descriptor(batch["segment"])
        proto_loss, usage = self.memory.loss(chi, batch["segment_mask"].all(1), self.history.identity.weight)
        return forecast_loss + proto_loss, usage


@dataclass
class DetectorState:
    observed: torch.Tensor
    normal: torch.Tensor

    def detach(self):
        return DetectorState(self.observed.detach(), self.normal.detach())


class DualTrajectoryDetector(nn.Module):
    def __init__(self, identity, variant="full", fixed_retention=0.9, sources=None):
        super().__init__()
        c, d = identity.shape
        self.variant = variant
        self.fixed_retention = fixed_retention
        self.sources = set(["relation", "marginal", "joint"] if sources is None else sources)
        self.register_buffer("identity", identity.detach().clone())
        self.evidence = mlp(79, 64, 32)
        self.normal_context = mlp(76 + d, 64, 32)
        self.gate = nn.Linear(96, 32)
        nn.init.constant_(self.gate.bias, 1)
        self.adapter = mlp(39 + d, 64, 32)
        self.gru = nn.GRUCell(32, 32)
        self.normal_gru = nn.GRUCell(32, 32) if variant == "independent_gru" else None
        self.head = mlp(96, 32, 1)

    def initial_state(self, batch, device):
        z = torch.zeros(batch, len(self.identity), 32, device=device)
        return DetectorState(z, z.clone())

    def step(self, f: dict, state: DetectorState | None = None, diagnostics=False):
        chi, prediction = f["chi"].float(), f["prediction"].float()
        batch, channels, _ = chi.shape
        state = state or self.initial_state(batch, chi.device)
        rm, rj = f["marginal"].float(), f["joint"].float()
        mv, jv = f["m_valid"].bool(), f["j_valid"].bool()
        rv = torch.full_like(mv, channels > 1 and "relation" in self.sources)
        if "marginal" not in self.sources:
            mv = torch.zeros_like(mv)
        if "joint" not in self.sources:
            jv = torch.zeros_like(jv)
        rm, rj = rm * mv[..., None], rj * jv[..., None]
        eps = f["residual"].float().clone()
        eps[..., 1] = eps[..., 1] * rv
        prediction = prediction.clone()
        prediction[..., 1] = prediction[..., 1] * rv
        availability = torch.stack([rv, mv, jv], -1).float()
        evidence = self.evidence(torch.cat([eps, (chi - rm) * mv[..., None],
                                            (chi - rj) * jv[..., None], availability], -1))
        identity = self.identity[None].expand(batch, -1, -1)
        context = self.normal_context(torch.cat([rm, rj, identity, availability[..., 1:]], -1))
        retain = torch.sigmoid(self.gate(torch.cat([evidence, context, state.observed - state.normal], -1)))
        if self.variant in {"no_alignment", "single_gru"}:
            retain = torch.ones_like(retain)
        elif self.variant == "fixed_gate":
            retain = torch.full_like(retain, self.fixed_retention)
        reference = torch.where(jv[..., None], rj, rm)
        old_normal = state.normal
        alignment_target = torch.zeros_like(old_normal) if self.variant == "zero_alignment" else old_normal
        aligned = retain * state.observed + (1 - retain) * alignment_target
        on = self.adapter(torch.cat([chi, prediction, identity], -1)).reshape(-1, 32)
        nn_input = self.adapter(torch.cat([reference, prediction, identity], -1)).reshape(-1, 32)
        new_o = self.gru(on, aligned.reshape(-1, 32)).reshape(batch, channels, 32)
        normal_cell = self.normal_gru if self.normal_gru is not None else self.gru
        new_n = normal_cell(nn_input, old_normal.reshape(-1, 32)).reshape(batch, channels, 32)
        usable = (mv | jv) & f["observed"].bool()
        new_o = new_o * usable[..., None]
        new_n = new_n * usable[..., None]
        if self.variant == "single_gru":
            new_n = torch.zeros_like(new_n)
        if self.variant == "mlp":
            new_o, new_n = torch.zeros_like(new_o), torch.zeros_like(new_n)
        delta = new_o - new_n
        logits = self.head(torch.cat([evidence, delta, new_n], -1)).squeeze(-1)
        scorable = f["observed"].bool() & f["prediction_ready"].bool()
        masked = logits.masked_fill(~scorable, -torch.inf)
        score, attribution = masked.max(-1)
        valid = scorable.any(-1)
        # Finite placeholder only inside the computation graph. Callers persist NaN for invalid outputs.
        score = torch.where(valid, score, torch.zeros_like(score))
        output = {"score": score, "valid": valid, "attribution": attribution}
        if diagnostics:
            output.update(retention=retain.mean(-1), state_difference=delta.norm(dim=-1),
                          marginal_difference=((chi - rm) * mv[..., None]).norm(dim=-1),
                          joint_difference=((chi - rj) * jv[..., None]).norm(dim=-1))
        return output, DetectorState(new_o, new_n)

    def forward_block(self, f: dict, state: DetectorState | None = None, diagnostics=False):
        """Batch the time-independent networks while retaining causal state updates."""
        chi, prediction = f["chi"].float(), f["prediction"].float()
        batch, length, channels, _ = chi.shape
        if length == 0:
            raise ValueError("A detector block must contain at least one time point.")
        state = state or self.initial_state(batch, chi.device)
        rm, rj = f["marginal"].float(), f["joint"].float()
        mv, jv = f["m_valid"].bool(), f["j_valid"].bool()
        rv = torch.full_like(mv, channels > 1 and "relation" in self.sources)
        if "marginal" not in self.sources:
            mv = torch.zeros_like(mv)
        if "joint" not in self.sources:
            jv = torch.zeros_like(jv)
        rm, rj = rm * mv[..., None], rj * jv[..., None]
        eps = f["residual"].float().clone()
        eps[..., 1] *= rv
        prediction = prediction.clone()
        prediction[..., 1] *= rv
        availability = torch.stack([rv, mv, jv], -1).float()
        evidence = self.evidence(torch.cat([eps, (chi - rm) * mv[..., None],
                                            (chi - rj) * jv[..., None], availability], -1))
        identity = self.identity[None, None].expand(batch, length, -1, -1)
        context = self.normal_context(torch.cat([rm, rj, identity, availability[..., 1:]], -1))
        reference = torch.where(jv[..., None], rj, rm)
        on = self.adapter(torch.cat([chi, prediction, identity], -1))
        normal_input = self.adapter(torch.cat([reference, prediction, identity], -1))
        usable = (mv | jv) & f["observed"].bool()
        # Unbind once: repeated select-backward operations otherwise clear a full
        # time-sized gradient buffer at every recurrent step.
        evidence_steps = evidence.transpose(0, 1).contiguous().unbind(0)
        context_steps = context.transpose(0, 1).contiguous().unbind(0)
        on_steps = on.transpose(0, 1).contiguous().reshape(length, -1, 32).unbind(0)
        normal_steps = normal_input.transpose(0, 1).contiguous().reshape(length, -1, 32).unbind(0)
        usable_steps = usable.transpose(0, 1).unbind(0)
        observed_states, normal_states, retentions = [], [], []
        old_o, old_n = state.observed, state.normal
        for t in range(length):
            retain = torch.sigmoid(self.gate(torch.cat([evidence_steps[t], context_steps[t], old_o - old_n], -1)))
            if self.variant in {"no_alignment", "single_gru"}:
                retain = torch.ones_like(retain)
            elif self.variant == "fixed_gate":
                retain = torch.full_like(retain, self.fixed_retention)
            target = torch.zeros_like(old_n) if self.variant == "zero_alignment" else old_n
            aligned = retain * old_o + (1 - retain) * target
            if self.normal_gru is None:
                inputs = torch.cat([on_steps[t], normal_steps[t]], 0)
                previous = torch.cat([aligned.reshape(-1, 32), old_n.reshape(-1, 32)], 0)
                new_o, new_n = self.gru(inputs, previous).chunk(2, 0)
                new_o, new_n = new_o.reshape(batch, channels, 32), new_n.reshape(batch, channels, 32)
            else:
                new_o = self.gru(on_steps[t], aligned.reshape(-1, 32)).reshape(batch, channels, 32)
                new_n = self.normal_gru(normal_steps[t], old_n.reshape(-1, 32)).reshape(batch, channels, 32)
            new_o, new_n = new_o * usable_steps[t][..., None], new_n * usable_steps[t][..., None]
            if self.variant == "single_gru":
                new_n = torch.zeros_like(new_n)
            if self.variant == "mlp":
                new_o, new_n = torch.zeros_like(new_o), torch.zeros_like(new_n)
            observed_states.append(new_o)
            normal_states.append(new_n)
            if diagnostics:
                retentions.append(retain)
            old_o, old_n = new_o, new_n
        observed, normal = torch.stack(observed_states, 1), torch.stack(normal_states, 1)
        delta = observed - normal
        logits = self.head(torch.cat([evidence, delta, normal], -1)).squeeze(-1)
        scorable = f["observed"].bool() & f["prediction_ready"].bool()
        score, attribution = logits.masked_fill(~scorable, -torch.inf).max(-1)
        valid = scorable.any(-1)
        output = {"score": torch.where(valid, score, torch.zeros_like(score)),
                  "valid": valid, "attribution": attribution}
        if diagnostics:
            output.update(retention=torch.stack(retentions, 1).mean(-1), state_difference=delta.norm(dim=-1),
                          marginal_difference=((chi - rm) * mv[..., None]).norm(dim=-1),
                          joint_difference=((chi - rj) * jv[..., None]).norm(dim=-1))
        return output, DetectorState(old_o, old_n)
