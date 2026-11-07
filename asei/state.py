"""Shared contractive normal dynamics."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from typing import List


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


@torch.jit.script
def normal_scan(inputs: torch.Tensor, state: torch.Tensor, valid: torch.Tensor,
                weight: torch.Tensor, bias: torch.Tensor, eta: float):
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for t in range(inputs.size(1)):
        state = (1 - eta) * state + eta * torch.tanh(F.linear(state, weight, bias) + inputs[:, t])
        state = state.masked_fill(~valid[:, t, :, None], 0)
        outputs.append(state)
    return torch.stack(outputs, 1), state


@torch.jit.script
def observed_scan(inputs: torch.Tensor, normals: torch.Tensor, previous_normal: torch.Tensor,
                  observed: torch.Tensor, gate_base: torch.Tensor, gate_delta: torch.Tensor,
                  valid: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                  eta: float, mode: int, fixed: float):
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    gates = torch.jit.annotate(List[torch.Tensor], [])
    for t in range(inputs.size(1)):
        delta = observed - previous_normal
        if mode == 1:
            retention = torch.ones_like(delta)
        elif mode == 2:
            retention = torch.zeros_like(delta)
        elif mode == 3:
            retention = torch.full_like(delta, fixed)
        else:
            retention = torch.sigmoid(gate_base[:, t] + F.linear(delta, gate_delta))
        aligned = previous_normal + retention * delta
        observed = (1 - eta) * aligned + eta * torch.tanh(F.linear(aligned, weight, bias) + inputs[:, t])
        observed = observed.masked_fill(~valid[:, t, :, None], 0)
        previous_normal = normals[:, t]
        outputs.append(observed)
        gates.append(retention)
    return torch.stack(outputs, 1), torch.stack(gates, 1), observed
