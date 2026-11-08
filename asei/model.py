"""Frozen normal references and two live trajectories with learned alignment."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .state import normal_scan, observed_scan
from .resources import cooperate


@dataclass
class TrajectoryState:
    normal: torch.Tensor
    observed: torch.Tensor

    def detach(self):
        return TrajectoryState(self.normal.detach(), self.observed.detach())


class DualTrajectoryDetector(nn.Module):
    def __init__(self, normal, settings):
        super().__init__()
        from .settings import VARIANTS
        self.normal = normal.freeze()
        self.settings = dict(settings)
        self.variant = settings.get("variant", "full")
        if self.variant not in VARIANTS:
            raise ValueError("Unsupported detector variant: " + self.variant)
        embedding = int(settings.get("channel_embedding_dim", 0))
        self.identity = nn.Embedding(normal.schema.channels, embedding) if embedding else None
        self.evidence_transform = settings.get("evidence_transform", "identity")
        hidden = int(settings.get("evidence_hidden", 64))
        readout = int(settings.get("readout_hidden", 32))
        self.evidence = nn.Sequential(nn.Linear(90 + embedding, hidden), nn.GELU(), nn.Linear(hidden, 32))
        self.gate = nn.Linear(96, 32)
        nn.init.constant_(self.gate.bias, 1.)
        self.readout = nn.Sequential(nn.Linear(96, readout), nn.GELU(), nn.Linear(readout, 1))
        self.single = nn.GRUCell(32, 32) if self.variant == "single_gru" else None

    def train(self, mode=True):
        super().train(mode)
        self.normal.eval()
        return self

    def initial_state(self, batch):
        zeros = self.normal.schema.amp_mean.new_zeros(batch, self.normal.schema.channels, 32)
        return TrajectoryState(zeros, zeros.clone())

    def _reference_replay(self, f, state, training=False):
        """Only the previous normal state and available peers can select a reference."""
        teacher, memory = self.normal.teacher, self.normal.memory
        tables = teacher.prototype_inputs(memory)
        if not training and "reference_index" in f:
            reference, pair, available, external = memory.lookup_reference(f["reference_index"])
            adapted, projected = teacher.reference_input(f["reference_index"], tables)
            valid = f["sv"] & available
            states, state = normal_scan(projected, state, valid,
                teacher.cell.hidden.weight, teacher.cell.hidden.bias, teacher.cell.eta)
            return dict(reference=reference, pair=pair, available=available, external=external,
                        reference_index=f["reference_index"], peer_cost=f["peer_cost"], adapted=adapted, states=states), state
        collected = {name: [] for name in ("reference", "pair", "available", "external", "reference_index", "peer_cost", "adapted", "states")}
        chunk = self.normal.config["memory"]["reference_block_length"]
        for begin in range(0, f["v"].shape[1], chunk):
            cooperate()
            end = min(begin + chunk, f["v"].shape[1])
            block = {k: v[:, begin:end] for k, v in f.items()}
            with torch.no_grad():
                prepared = memory.prepare_peers(block, self.settings.get("reference_trim", .1))
            for t in range(begin, end):
                step = {k: v[:, t] for k, v in f.items()}
                with torch.no_grad():
                    prior, logits = teacher.prior(state.detach())
                    reference, pair, available, external, index, peer_cost = memory.reference(step, prior, logits,
                        self.settings.get("reference_trim", .1), self.settings.get("reference_prior", .25),
                        external=training or self.variant != "no_external",
                        prepared={k: v[:, t - begin] for k, v in prepared.items()}, return_index=True, return_support=True)
                adapted, projected = teacher.reference_input(index, tables)
                valid = step["sv"] & available
                state = teacher.cell.forward_projected(projected.reshape(-1, 32), state.reshape(-1, 32)).reshape_as(state)
                state = state.masked_fill(~valid[..., None], 0)
                values = (reference, pair, available, external, index, peer_cost, adapted, state)
                for name, value in zip(collected, values):
                    collected[name].append(value)
        return {k: torch.stack(v, 1) for k, v in collected.items()}, state

    def teacher_block(self, f, state=None, closed_loop=False):
        teacher = self.normal.teacher
        state = self.initial_state(f["v"].shape[0]).normal if state is None else state
        if closed_loop:
            values, state = self._reference_replay(f, state, training=True)
            return values["states"], state
        adapted = teacher.adapt(f["v"], f["pair"])
        return normal_scan(teacher.cell.input(adapted), state, f["sv"],
                           teacher.cell.hidden.weight, teacher.cell.hidden.bias, teacher.cell.eta)

    @torch.no_grad()
    def frozen_block(self, f, state=None):
        """Prepare frozen evidence once per block; all detector gradients remain live."""
        normal, teacher = self.normal, self.normal.teacher
        state = self.initial_state(f["v"].shape[0]).normal if state is None else state
        references, final = self._reference_replay(f, state)
        reference, pair = references["reference"], references["pair"]
        rv, external = references["available"], references["external"]
        long = normal.memory.long_evidence(f)
        # The marginal query depends only on current descriptors, not on the reference.
        marginal = normal.memory.short_marginal(f)
        distances, available = normal.memory.short_evidence(dict(f, short_marginal=marginal), reference, pair, rv, external,
            peer_cost=references["peer_cost"] if self.settings.get("reference_guard_multiplier", 0.) else None)
        short_energy = (distances / normal.memory.energy_scale[:, 0] * available).sum(-1) / available.sum(-1).clamp(min=1)
        lavail = torch.stack([long["mv"], long["jv"]], -1)
        long_energy = (long["long_dist"] / normal.memory.energy_scale[:, 1] * lavail).sum(-1) / lavail.sum(-1).clamp(min=1)
        difference = f["v"] - reference
        if normal.schema.cat_indices:
            ids = normal.schema.cat_indices
            difference[..., ids, :] = (f["pair"][..., ids, :] != pair[..., ids, :]).float()
        valid_state = f["sv"] & rv
        masks = torch.stack([f["sv"], f["lv"], f["forecast_ready"][..., 0], f["forecast_ready"][..., 1],
                             long["mv"], long["jv"], rv, external], -1).float()
        feature = torch.cat([f["forecast"].masked_fill(~f["forecast_ready"], 0),
                             difference.masked_fill(~valid_state[..., None], 0), long["md"], long["jd"],
                             short_energy.log1p()[..., None], long_energy.log1p()[..., None], masks], -1)
        return dict(feature=feature, reference=reference, reference_pair=pair,
            reference_input=references["adapted"], normal=references["states"], final_normal=final,
            actual_projected=teacher.cell.input(teacher.adapt(f["v"], f["pair"])), reference_valid=valid_state,
            short_energy=short_energy, short_distance=distances, short_available=available,
            long_distance=long["long_dist"], long_available=lavail,
            reference_index=references["reference_index"], m_index=long["m_index"],
            j_index=long["j_index"], short_marginal=marginal, peer_cost=references["peer_cost"], external=external)

    def encode_evidence(self, features):
        if self.evidence_transform == "signed_log1p":
            # Compress outliers without discarding sign or magnitude. The two
            # energy coordinates are already log-transformed; masks stay binary.
            numeric = features[..., :80]
            features = torch.cat([numeric.sign() * numeric.abs().log1p(), features[..., 80:]], -1)
        if self.identity is not None:
            identity = self.identity.weight.expand(*features.shape[:-2], -1, -1)
            features = torch.cat([features, identity], -1)
        return self.evidence(features)

    def forward_block(self, f, state=None, return_aux=False, prepared=None):
        state = self.initial_state(f["v"].shape[0]) if state is None else state
        prepared = self.frozen_block(f, state.normal) if prepared is None else prepared
        b = self.encode_evidence(prepared["feature"])
        normals, valid = prepared["normal"], prepared["reference_valid"]
        if self.variant in {"mlp", "energy_only"}:
            observed = normals
            retention = torch.zeros_like(normals)
            final_observed = normals[:, -1]
        elif self.single is not None:
            current, outputs = state.observed, []
            for t in range(b.shape[1]):
                current = self.single(b[:, t].reshape(-1, 32), current.reshape(-1, 32)).reshape_as(current)
                current = current.masked_fill(~valid[:, t, :, None], 0)
                outputs.append(current)
            observed, final_observed = torch.stack(outputs, 1), current
            retention = torch.zeros_like(observed)
        else:
            gate_base = F.linear(torch.cat([b, prepared["reference_input"]], -1), self.gate.weight[:, :64], self.gate.bias)
            mode = {"no_alignment": 1, "zero_alignment": 2, "fixed_gate": 3}.get(self.variant, 0)
            cell = self.normal.teacher.cell
            observed, retention, final_observed = observed_scan(prepared["actual_projected"], normals,
                state.normal, state.observed, gate_base, self.gate.weight[:, 64:], valid,
                cell.hidden.weight, cell.hidden.bias, cell.eta, mode, float(self.settings.get("fixed_retention", .9)))
        dynamic = self.readout(torch.cat([b, observed - normals, normals], -1)).squeeze(-1)
        dv = valid & f["target"] & (self.variant != "energy_only")
        sv = prepared["short_available"].any(-1) & f["target"] & (self.variant != "no_energy")
        short = prepared["short_energy"].log1p().masked_fill(~sv, float("-inf")).amax(-1)
        learned = dynamic.masked_fill(~dv, float("-inf")).amax(-1)
        output = dict(short=torch.where(sv.any(-1), short, torch.zeros_like(short)),
            dynamic=torch.where(dv.any(-1), learned, torch.zeros_like(learned)),
            short_valid=sv.any(-1), dynamic_valid=dv.any(-1),
            attribution_short=prepared["short_energy"].masked_fill(~sv, -1).argmax(-1),
            attribution_dynamic=dynamic.masked_fill(~dv, float("-inf")).argmax(-1))
        # One block check avoids synchronizing the GPU at every time point.
        if not torch.isfinite(torch.stack([output["short"], output["dynamic"]])).all():
            raise FloatingPointError("Non-finite scorable detector output.")
        if return_aux:
            output.update({k: prepared[k] for k in ("normal", "reference", "reference_pair", "reference_valid",
                "short_distance", "short_available", "long_distance", "long_available")})
            output.update(observed=observed, retention=retention, channel_dynamic=dynamic, channel_valid=dv)
        valid_channels = (f["sv"] & f["target"]).sum(-1).clamp(min=1)
        output["short_reference_coverage"] = (prepared["short_available"][..., 1] & f["target"]).sum(-1) / valid_channels
        return output, TrajectoryState(prepared["final_normal"], final_observed)

    def step(self, features, state=None, return_aux=False):
        output, state = self.forward_block({k: v[:, None] for k, v in features.items()}, state, return_aux)
        return {k: v[:, 0] for k, v in output.items()}, state

    @torch.no_grad()
    def routing(self, record, block_length=128, progress=None):
        """Compact normal-only retrieval identity; no trainable outputs are cached."""
        return self.routing_many([record], block_length, progress)[0]

    @torch.no_grad()
    def routing_many(self, records, block_length=128, progress=None):
        """Replay independent tasks together, preserving each task's own state boundary."""
        import numpy as np
        results, state = [{} for _ in records], None
        maximum = max(len(record.z) for record in records)
        for start in range(0, maximum, block_length):
            cooperate()
            f = self.normal.schema.block(records, [start] * len(records), min(block_length, maximum - start))
            values = self.frozen_block(f, state)
            state = values["final_normal"]
            for name in ("reference_index", "m_index", "j_index", "short_marginal", "peer_cost"):
                dtype = np.float32 if name in {"short_marginal", "peer_cost"} else np.int32
                array = values[name].cpu().numpy().astype(dtype)
                for i, record in enumerate(records):
                    count = min(block_length, len(record.z) - start)
                    if count > 0:
                        results[i].setdefault(name, []).append(array[i, :count].copy())
            if progress is not None:
                progress(min(start + block_length, maximum))
        return [{k: np.concatenate(v) for k, v in result.items()} for result in results]
