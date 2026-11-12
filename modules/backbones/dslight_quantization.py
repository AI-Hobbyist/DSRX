"""Opt-in row-wise INT8 weight storage for DSLight pointwise convolutions.

Execution dequantizes weights for the existing floating-point convolution backend.
This is not an integer activation/kernel implementation or a training format.
"""

import torch
from torch import nn
from torch.nn import functional as F

from modules.backbones.dslight import DSLightBackbone


class WeightOnlyInt8Pointwise(nn.Module):
    def __init__(self, source):
        super().__init__()
        weight = source.weight.detach().float()
        scale = weight.abs().flatten(1).amax(dim=1).clamp_min(1e-12) / 127
        packed = torch.round(weight / scale[:, None, None]).clamp(-127, 127)
        self.register_buffer("packed_weight", packed.to(torch.int8))
        self.register_buffer("scale", scale)
        self.register_buffer(
            "bias", source.bias.detach().clone() if source.bias is not None else None
        )
        self.eval()

    def forward(self, x):
        if self.training:
            raise RuntimeError("Weight-only DSLight quantization is inference-only")
        weight = self.packed_weight.to(x.dtype) * self.scale.to(x.dtype)[:, None, None]
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.conv1d(x, weight, bias)


def quantize_dslight_pointwise(model):
    """Replace block projections only; preserve depthwise/time/output paths as floats."""
    if model.training:
        raise ValueError("Call eval() before quantizing DSLight")
    converted = []
    for backbone_name, backbone in list(model.named_modules()):
        if not isinstance(backbone, DSLightBackbone):
            continue
        for index, block in enumerate(backbone.blocks):
            for name in ("expand", "project", "condition"):
                convolution = getattr(block, name)
                if isinstance(convolution, WeightOnlyInt8Pointwise):
                    continue
                if type(convolution) is not nn.Conv1d or convolution.kernel_size != (
                    1,
                ):
                    raise TypeError(f"Unsupported DSLight projection: {name}")
                setattr(block, name, WeightOnlyInt8Pointwise(convolution))
                prefix = f"{backbone_name}." if backbone_name else ""
                converted.append(f"{prefix}blocks.{index}.{name}")
    return converted


def load_weight_only_package(model, package):
    """Load into an already configured float model through an explicit format boundary."""
    if package.get("format") != "dslight-int8-weight-only-v1":
        raise ValueError("Unsupported DSLight weight-only package format")
    converted = quantize_dslight_pointwise(model)
    if converted != package["layers"]:
        raise ValueError("Quantized layer manifest does not match this architecture")
    model.load_state_dict(package["state_dict"], strict=True)
    return model
