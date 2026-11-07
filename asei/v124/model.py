"""Two live trajectories sharing fixed normal dynamics, with trainable alignment."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class State124:
    normal: torch.Tensor
    observed: torch.Tensor

    def detach(self):
        return State124(self.normal.detach(), self.observed.detach())


class Detector124(nn.Module):
    def __init__(self, normal, settings):
        super().__init__()
        self.normal = normal.freeze()
        self.settings = dict(settings)
        self.variant = settings.get("variant", "full")
        supported = {"full", "mlp", "single_gru", "no_alignment", "zero_alignment", "fixed_gate",
                     "no_energy", "energy_only", "no_tail", "no_state", "no_external"}
        if self.variant not in supported:
            raise ValueError("Unsupported in_1_2_4 variant: " + self.variant)
        self.evidence = nn.Sequential(nn.Linear(90, 64), nn.GELU(), nn.Linear(64, 32))
        self.gate = nn.Linear(96, 32)
        nn.init.constant_(self.gate.bias, 1.)
        self.readout = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
        self.single = nn.GRUCell(32, 32) if self.variant == "single_gru" else None

    def train(self, mode=True):
        super().train(mode)
        self.normal.eval()
        return self

    def initial_state(self, batch):
        size = (batch, self.normal.schema.channels, 32)
        zeros = self.normal.schema.amp_mean.new_zeros(size)
        return State124(zeros, zeros.clone())

    def teacher_block(self, f, state=None, closed_loop=False):
        """Training-only replay, also used for clean paired state targets."""
        teacher = self.normal.teacher
        if state is None:
            state = f["v"].new_zeros(f["v"].shape[0], self.normal.schema.channels, 32)
        output = []
        for t in range(f["v"].shape[1]):
            step = {k: v[:, t] for k, v in f.items()}
            valid = step["sv"]
            v, pair = step["v"], step["pair"]
            if closed_loop:
                with torch.no_grad():
                    prior, logits = teacher.prior(state.detach())
                    v, pair, available, _ = self.normal.memory.reference(step, prior, logits,
                        self.settings.get("reference_trim", .1), self.settings.get("reference_prior", .25))
                valid = valid & available
            state = teacher.cell(teacher.adapt(v, pair).reshape(-1, 32), state.reshape(-1, 32)).reshape_as(state)
            state = state.masked_fill(~valid[..., None], 0)
            output.append(state)
        return torch.stack(output, 1), state

    def forward_block(self, f, state=None, return_aux=False):
        state = self.initial_state(f["v"].shape[0]) if state is None else state
        with torch.no_grad():
            long = self.normal.memory.long_evidence(f)
        collected = []
        for t in range(f["v"].shape[1]):
            step = {k: v[:, t] for k, v in f.items()}
            context = {k: v[:, t] for k, v in long.items()}
            result, state = self.step(step, state, context, return_aux)
            collected.append(result)
        return {k: torch.stack([r[k] for r in collected], 1) for k in collected[0]}, state

    def step(self, f, state=None, long=None, return_aux=False):
        normal, teacher = self.normal, self.normal.teacher
        state = self.initial_state(f["v"].shape[0]) if state is None else state
        with torch.no_grad():
            if long is None:
                long = normal.memory.long_evidence(f)
            prior, logits = teacher.prior(state.normal)
            reference, pair, rv, external = normal.memory.reference(f, prior, logits,
                self.settings.get("reference_trim", .1), self.settings.get("reference_prior", .25),
                external=self.variant != "no_external")
            distances, available = normal.memory.short_evidence(f, reference, pair, rv, external)
            short_energy = (distances / normal.memory.energy_scale[:, 0] * available).sum(-1) / available.sum(-1).clamp(min=1)
            lavail = torch.stack([long["mv"], long["jv"]], -1)
            long_energy = (long["long_dist"] / normal.memory.energy_scale[:, 1] * lavail).sum(-1) / lavail.sum(-1).clamp(min=1)
            difference = f["v"] - reference
            if normal.schema.cat_indices:
                ids = normal.schema.cat_indices
                difference[..., ids, :] = (f["pair"][..., ids, :] != pair[..., ids, :]).float()
            valid_state = f["sv"] & rv
            ref_input = teacher.adapt(reference, pair)
            actual_input = teacher.adapt(f["v"], f["pair"])
            normal_state = teacher.cell(ref_input.reshape(-1, 32), state.normal.reshape(-1, 32)).reshape_as(state.normal)
            normal_state = normal_state.masked_fill(~valid_state[..., None], 0)
        masks = torch.stack([f["sv"], f["lv"], f["forecast_ready"][..., 0], f["forecast_ready"][..., 1],
                             long["mv"], long["jv"], rv, external], -1).float()
        feature = torch.cat([f["forecast"].masked_fill(~f["forecast_ready"], 0),
                             difference.masked_fill(~f["sv"][..., None], 0), long["md"], long["jd"],
                             short_energy.log1p()[..., None], long_energy.log1p()[..., None], masks], -1)
        b = self.evidence(feature)
        previous_delta = state.observed - state.normal
        retention = torch.sigmoid(self.gate(torch.cat([b, ref_input, previous_delta], -1)))
        if self.variant == "no_alignment":
            retention = torch.ones_like(retention)
        elif self.variant == "zero_alignment":
            retention = torch.zeros_like(retention)
        elif self.variant == "fixed_gate":
            retention = torch.full_like(retention, self.settings.get("fixed_retention", .9))
        aligned = state.normal + retention * previous_delta
        # Frozen weights remain differentiable with respect to alignment and prior O state.
        observed = teacher.cell(actual_input.reshape(-1, 32), aligned.reshape(-1, 32)).reshape_as(state.observed)
        if self.single is not None:
            observed = self.single(b.reshape(-1, 32), state.observed.reshape(-1, 32)).reshape_as(observed)
        elif self.variant == "mlp":
            observed = normal_state
        observed = observed.masked_fill(~valid_state[..., None], 0)
        delta = observed - normal_state
        dynamic = self.readout(torch.cat([b, delta, normal_state], -1)).squeeze(-1)
        dv = valid_state & f["target"]
        sv = available.any(-1) & f["target"]
        if self.variant == "no_energy":
            sv = torch.zeros_like(sv)
        if self.variant == "energy_only":
            dv = torch.zeros_like(dv)
        short = short_energy.log1p().masked_fill(~sv, float("-inf")).amax(-1)
        learned = dynamic.masked_fill(~dv, float("-inf")).amax(-1)
        out = dict(short=torch.where(sv.any(-1), short, torch.zeros_like(short)),
                   dynamic=torch.where(dv.any(-1), learned, torch.zeros_like(learned)),
                   short_valid=sv.any(-1), dynamic_valid=dv.any(-1),
                   attribution_short=short_energy.masked_fill(~sv, -1).argmax(-1),
                   attribution_dynamic=dynamic.masked_fill(~dv, float("-inf")).argmax(-1))
        if not torch.isfinite(out["short"]).all() or not torch.isfinite(out["dynamic"]).all():
            raise FloatingPointError("Non-finite scorable in_1_2_4 output.")
        if return_aux:
            out.update(normal=normal_state, observed=observed, retention=retention, reference=reference,
                       reference_pair=pair, reference_valid=valid_state, short_distance=distances,
                       short_available=available, long_distance=long["long_dist"], long_available=lavail)
        return out, State124(normal_state, observed)
