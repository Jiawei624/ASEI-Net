"""Shared recurrent dynamics and the training-only normal-state teacher."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ContractiveCell(nn.Module):
    def __init__(self, size=32, eta=0.2, rho=0.95):
        super().__init__()
        if not 0 < eta <= 1 or not 0 < rho < 1:
            raise ValueError("Contractive dynamics require eta in (0,1] and rho in (0,1).")
        self.eta, self.rho = float(eta), float(rho)
        self.hidden = nn.Linear(size, size)
        self.input = nn.Linear(size, size, bias=False)
        self.project()

    def forward(self, inputs, previous):
        return self.forward_projected(self.input(inputs), previous)

    def forward_projected(self, projected_input, previous):
        return (1 - self.eta) * previous + self.eta * torch.tanh(self.hidden(previous) + projected_input)

    @torch.no_grad()
    def project(self):
        u, values, vh = torch.linalg.svd(self.hidden.weight, full_matrices=False)
        self.hidden.weight.copy_((u * values.clamp(max=self.rho * (1 - 1e-6))) @ vh)
        # TF32 matrix products can round the reconstructed matrix outside the
        # spectral ball. Check the stored FP32 matrix and correct that drift.
        norm = torch.linalg.svdvals(self.hidden.weight)[0]
        self.hidden.weight.mul_((self.rho * (1 - 1e-6) / norm.clamp_min(1e-12)).clamp(max=1))

    @property
    def contraction(self):
        return 1 - self.eta + self.eta * self.rho


def recurrent_cell(kind, eta=0.2, rho=0.95):
    if kind == "contractive":
        return ContractiveCell(32, eta, rho)
    if kind == "gru":
        return nn.GRUCell(32, 32)
    raise ValueError("Unknown recurrent cell: " + str(kind))


class NormalStateTeacher(nn.Module):
    """Reads only normal references/predictions; future descriptions are targets."""

    def __init__(self, identity, cell="contractive", eta=0.2, rho=0.95, sources=None):
        super().__init__()
        self.sources = set(["relation", "marginal", "joint"] if sources is None else sources)
        self.register_buffer("identity", identity.detach().clone())
        self.adapter = nn.Sequential(nn.Linear(39 + identity.shape[-1], 64), nn.GELU(), nn.Linear(64, 32))
        self.gru = recurrent_cell(cell, eta, rho)
        self.current_head = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 37))
        self.next_head = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 37))

    def forward_block(self, features, state=None):
        prediction = features["prediction"].float().clone()
        if "relation" not in self.sources or len(self.identity) < 2:
            prediction[..., 1] = 0
        mv = features["m_valid"].bool() & ("marginal" in self.sources)
        jv = features["j_valid"].bool() & ("joint" in self.sources)
        reference = torch.where(jv[..., None], features["joint"], features["marginal"] * mv[..., None])
        usable = (mv | jv) & features["observed"].bool() & features["prediction_ready"].bool()
        batch, length, channels, _ = reference.shape
        identity = self.identity[None, None].expand(batch, length, -1, -1)
        inputs = self.adapter(torch.cat([reference, prediction, identity], -1))
        if isinstance(self.gru, ContractiveCell):
            inputs = self.gru.input(inputs)
        inputs = inputs.transpose(0, 1).contiguous().unbind(0)
        masks = usable.transpose(0, 1).unbind(0)
        if state is None:
            state = reference.new_zeros(batch, channels, 32)
        outputs = []
        for inputs_t, valid_t in zip(inputs, masks):
            update = self.gru.forward_projected if isinstance(self.gru, ContractiveCell) else self.gru
            state = update(inputs_t.reshape(-1, 32), state.reshape(-1, 32)).reshape(batch, channels, 32)
            state = state * valid_t[..., None]
            outputs.append(state)
        return torch.stack(outputs, 1), state, usable

    @torch.no_grad()
    def project(self):
        if isinstance(self.gru, ContractiveCell):
            self.gru.project()


def balanced_description_loss(predicted, target):
    errors = F.huber_loss(predicted, target, reduction="none")
    return .5 * (errors[..., :5].mean(-1) + errors[..., 5:].mean(-1))
