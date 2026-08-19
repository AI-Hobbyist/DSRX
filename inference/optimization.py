"""Runtime optimizations shared by command-line and backend inference.

The defaults in this module are intentionally inference-only.  Large linear
layers use FP16 Tensor Core math while the rest of the model and all public
inputs/outputs stay in FP32.  Rectified-flow velocity backbones are traced and
frozen once, then warmed with representative time-axis lengths.
"""

from __future__ import annotations

import gc
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hparams import hparams


DEFAULT_WARMUP_FRAMES = (128, 512, 2048, 6000, 10000)
DEFAULT_MIN_LINEAR_ELEMENTS = 16_384


class SelectiveFP16Linear(nn.Module):
    """Inference-only Linear with FP16-resident weights and FP32 boundaries."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer("weight", linear.weight.detach().half().clone())
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().half().clone(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        return F.linear(x.half(), self.weight, self.bias).to(input_dtype)


@dataclass
class BackboneOptimization:
    name: str
    traced: bool = False
    trace_seconds: float = 0.0
    warmup_frames: List[int] = field(default_factory=list)
    warmup_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class InferenceOptimizationReport:
    model_kind: str
    enabled: bool
    device: str
    reason: Optional[str] = None
    tf32: bool = False
    selective_fp16: bool = False
    min_linear_elements: int = DEFAULT_MIN_LINEAR_ELEMENTS
    converted_linears: int = 0
    converted_weight_elements: int = 0
    converted_weight_bytes_fp32: int = 0
    torchscript: bool = False
    backbones: List[BackboneOptimization] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _cuda_device(device) -> Optional[torch.device]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def apply_inference_math_mode(model_kind: str, device) -> None:
    """Apply per-model global CUDA math settings before every forward."""

    if _cuda_device(device) is None:
        return
    acoustic = model_kind == "acoustic"
    torch.backends.cuda.matmul.allow_tf32 = acoustic
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high" if acoustic else "highest")


def _replace_large_linears(model: nn.Module, minimum_elements: int) -> tuple[int, int]:
    converted = 0
    converted_elements = 0

    def visit(parent: nn.Module) -> None:
        nonlocal converted, converted_elements
        for name, child in list(parent.named_children()):
            if isinstance(child, SelectiveFP16Linear):
                continue
            if isinstance(child, nn.Linear) and child.weight.numel() >= minimum_elements:
                converted += 1
                converted_elements += child.weight.numel()
                setattr(parent, name, SelectiveFP16Linear(child))
            else:
                visit(child)

    visit(model)
    return converted, converted_elements


def _velocity_backbones(model: nn.Module):
    for name, module in model.named_modules():
        velocity_fn = getattr(module, "velocity_fn", None)
        if isinstance(velocity_fn, nn.Module):
            yield name, module, velocity_fn


def _parse_warmup_frames(value) -> Sequence[int]:
    if value is None:
        return DEFAULT_WARMUP_FRAMES
    if isinstance(value, str):
        value = [part.strip() for part in value.split("|")]
    result = []
    for item in value:
        frames = int(item)
        if frames > 0 and frames not in result:
            result.append(frames)
    return result or DEFAULT_WARMUP_FRAMES


def _trace_and_warm_backbone(
    *,
    name: str,
    owner: nn.Module,
    velocity_fn: nn.Module,
    device: torch.device,
    warmup_frames: Sequence[int],
) -> BackboneOptimization:
    result = BackboneOptimization(name=name)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    try:
        num_feats = int(owner.num_feats)
        out_dims = int(owner.out_dims)
        hidden_size = int(hparams["hidden_size"])
        trace_frames = int(warmup_frames[min(1, len(warmup_frames) - 1)])
        example_spec = torch.randn(
            1, num_feats, out_dims, trace_frames, device=device
        )
        example_step = torch.tensor([500.0], device=device)
        example_cond = torch.randn(1, hidden_size, trace_frames, device=device)

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        traced = torch.jit.trace(
            velocity_fn,
            (example_spec, example_step, example_cond),
            check_trace=False,
            strict=False,
        )
        traced = torch.jit.freeze(traced.eval())
        setattr(owner, "velocity_fn", traced)
        torch.cuda.synchronize(device)
        result.trace_seconds = time.perf_counter() - started
        result.traced = True

        del example_spec, example_step, example_cond
        started = time.perf_counter()
        with torch.inference_mode():
            for frames in warmup_frames:
                spec = torch.randn(1, num_feats, out_dims, frames, device=device)
                step = torch.tensor([500.0], device=device)
                cond = torch.randn(1, hidden_size, frames, device=device)
                traced(spec, step, cond)
                torch.cuda.synchronize(device)
                result.warmup_frames.append(int(frames))
                del spec, step, cond
        result.warmup_seconds = time.perf_counter() - started
    except Exception as exc:
        if result.traced:
            setattr(owner, "velocity_fn", velocity_fn)
            result.traced = False
        result.error = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"Inference TorchScript optimization failed for {name}: {result.error}. "
            "Continuing with the eager backbone.",
            RuntimeWarning,
        )
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
    return result


def optimize_model_for_inference(
    model: nn.Module,
    *,
    model_kind: str,
    device,
) -> InferenceOptimizationReport:
    """Apply the profiled DSRX inference defaults in-place."""

    if model_kind not in {"acoustic", "variance"}:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    resolved = _cuda_device(device)
    enabled = bool(hparams.get("inference_optimization", True))
    report = InferenceOptimizationReport(
        model_kind=model_kind,
        enabled=enabled,
        device=str(device),
    )
    if not enabled:
        report.reason = "disabled by inference_optimization"
        return report
    if resolved is None:
        report.reason = "CUDA is unavailable; keeping the original FP32 model"
        return report

    apply_inference_math_mode(model_kind, resolved)
    report.tf32 = model_kind == "acoustic"

    minimum_elements = int(
        hparams.get("inference_selective_fp16_min_elements", DEFAULT_MIN_LINEAR_ELEMENTS)
    )
    report.min_linear_elements = minimum_elements
    if bool(hparams.get("inference_selective_fp16", True)):
        converted, elements = _replace_large_linears(model, minimum_elements)
        report.selective_fp16 = converted > 0
        report.converted_linears = converted
        report.converted_weight_elements = elements
        report.converted_weight_bytes_fp32 = elements * 4

    if bool(hparams.get("inference_torchscript", True)):
        warmup_frames = _parse_warmup_frames(
            hparams.get("inference_warmup_frames", DEFAULT_WARMUP_FRAMES)
        )
        targets = list(_velocity_backbones(model))
        for name, owner, velocity_fn in targets:
            report.backbones.append(
                _trace_and_warm_backbone(
                    name=name,
                    owner=owner,
                    velocity_fn=velocity_fn,
                    device=resolved,
                    warmup_frames=warmup_frames,
                )
            )
        report.torchscript = bool(report.backbones) and all(
            item.traced for item in report.backbones
        )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(resolved)
    print(
        "| inference optimization: "
        f"kind={model_kind}, selective_fp16={report.converted_linears} linears, "
        f"torchscript={sum(item.traced for item in report.backbones)}/"
        f"{len(report.backbones)}, tf32={report.tf32}"
    )
    return report
