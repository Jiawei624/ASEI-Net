"""API compatibility with PyTorch >= 1.11."""
from __future__ import annotations

import inspect

import torch
from torch import nn


def transformer_encoder(layer, layers, norm):
    kwargs = {"norm": norm}
    if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
        kwargs["enable_nested_tensor"] = False
    return nn.TransformerEncoder(layer, layers, **kwargs)


def load_torch_checkpoint(path, map_location=None):
    """Load own/trusted checkpoints; request restricted loading wherever supported.

    PyTorch 1.11 does not expose weights_only. Do not retry a failed restricted
    load with unrestricted loading on newer PyTorch versions.
    """
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)
