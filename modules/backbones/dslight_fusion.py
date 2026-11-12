"""Explicit modern-PyTorch gate/projection compilation; normal export stays eager."""

import torch
from torch import nn
from torch.nn import functional as F

from modules.backbones.dslight import DSLightBackbone


def gated_projection(a, b, weight, bias):
    features = torch.cat((a * torch.sigmoid(b), a), dim=1).transpose(1, 2)
    return F.linear(features, weight.squeeze(-1), bias).transpose(1, 2)


def enable_dslight_fusion(model):
    """Compile the gate/input fusion explicitly; never install or select a fallback."""
    if model.training:
        raise ValueError("Enable DSLight fusion only after eval()")
    if not hasattr(torch, "compile"):
        raise RuntimeError("DSLight fusion requires a torch.compile-capable runtime")
    blocks = [
        block
        for module in model.modules()
        if isinstance(module, DSLightBackbone)
        for block in module.blocks
    ]
    if any(type(block.project) is not nn.Conv1d for block in blocks):
        raise TypeError("Compiled fusion requires floating pointwise projections")
    compiled = torch.compile(gated_projection, fullgraph=True, dynamic=True)
    for block in blocks:
        block.fused_projection = compiled
    return len(blocks)
