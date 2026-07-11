"""
Precision management utilities for DSRX.

Supports:
  - fp32   : Full precision (baseline)
  - fp16   : Half precision (CUDA AMP)
  - bf16   : BFloat16 precision (CUDA AMP, Ampere+)
  - int8   : Static/dynamic quantization (QAT + PTQ)

Provides:
  - PrecisionContext: unified dtype resolution & autocast wrapper
  - QAT utilities: prepare_qat / convert_qat / fuse_modules
  - ONNX quantization helpers (model dtype conversion, calibration)
"""

from __future__ import annotations

import contextlib
import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PRECISIONS = {'fp32', 'fp16', 'bf16', 'int8'}
VALID_TRAIN_PRECISIONS = {'32-true', '16-mixed', 'bf16-mixed'}
VALID_EXPORT_PRECISIONS = {'fp32', 'fp16', 'bf16', 'int8', 'all'}

TORCH_DTYPE_MAP: Dict[str, torch.dtype] = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

ONNX_DTYPE_MAP = {
    'fp32': 1,   # TensorProto.FLOAT
    'fp16': 10,  # TensorProto.FLOAT16
    'bf16': 16,  # TensorProto.BFLOAT16
    'int8': 3,   # TensorProto.INT8
}


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------

def resolve_dtype(dtype_spec: Union[str, torch.dtype, None]) -> torch.dtype:
    """Resolve a string precision name to a torch.dtype."""
    if dtype_spec is None:
        return torch.float32
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    dtype_spec = dtype_spec.lower()
    if dtype_spec not in TORCH_DTYPE_MAP:
        raise ValueError(
            f"Unknown precision '{dtype_spec}'. "
            f"Valid options: {sorted(TORCH_DTYPE_MAP.keys())}"
        )
    return TORCH_DTYPE_MAP[dtype_spec]


def supports_bf16() -> bool:
    """Check if the current CUDA device supports bfloat16."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def supports_fp16() -> bool:
    """Check if the current CUDA device supports float16."""
    return torch.cuda.is_available()


def auto_dtype(device: torch.device) -> torch.dtype:
    """Pick the best available dtype for the device."""
    if device.type == 'cuda':
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# PrecisionContext — unified autocast + dtype manager
# ---------------------------------------------------------------------------

class PrecisionContext:
    """Unified precision manager for training, inference, and export.

    Usage::

        ctx = PrecisionContext('fp16')
        model = ctx.convert_model(model)
        with ctx.autocast():
            output = model(x)

    During inference, you can call ``ctx.convert_model(model)`` once
    and then repeatedly call ``ctx.autocast()`` around each forward.
    """

    def __init__(
        self,
        precision: str = 'fp32',
        device_type: str = 'cuda',
        enabled: bool = True,
    ):
        if precision not in VALID_PRECISIONS and precision not in VALID_TRAIN_PRECISIONS:
            raise ValueError(
                f"Unknown precision '{precision}'. "
                f"Valid: {sorted(VALID_PRECISIONS | VALID_TRAIN_PRECISIONS)}"
            )

        # Normalise Lightning-style precision names
        _normalised = {
            '32-true': 'fp32',
            '16-mixed': 'fp16',
            'bf16-mixed': 'bf16',
        }
        self.name = _normalised.get(precision, precision)
        self.dtype = TORCH_DTYPE_MAP.get(self.name, torch.float32)
        self.device_type = device_type
        self.enabled = enabled and (self.name != 'fp32')

        # BFloat16 requires CUDA capability >= 8.0 (Ampere)
        if self.name == 'bf16' and not supports_bf16():
            import warnings
            warnings.warn(
                "bf16 requested but not supported by current device. "
                "Falling back to fp16."
            )
            self.name = 'fp16'
            self.dtype = torch.float16

    def convert_model(self, model: nn.Module) -> nn.Module:
        """Convert model parameters to the target dtype (in-place)."""
        if not self.enabled:
            return model
        return model.to(dtype=self.dtype)

    @contextlib.contextmanager
    def autocast(self):
        """Context-manager that enables AMP autocast for the target dtype."""
        if not self.enabled:
            yield
            return
        with torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=True,
        ):
            yield

    def __repr__(self) -> str:
        return (
            f"PrecisionContext(precision='{self.name}', "
            f"dtype={self.dtype}, enabled={self.enabled})"
        )


# ---------------------------------------------------------------------------
# QAT (Quantization-Aware Training) utilities
# ---------------------------------------------------------------------------

# Default layers that should NOT be quantized
_QAT_EXCLUDE_TYPES = (
    nn.Embedding,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
)

_QAT_FUSE_PATTERNS = [
    # (pattern, fuse_fn_name)
    (['conv1d', 'bn', 'relu'], 'fuse_conv_bn_relu'),
    (['conv1d', 'bn'], 'fuse_conv_bn'),
    (['linear', 'relu'], 'fuse_linear_relu'),
]


def _module_name_matches(name: str, patterns: List[str]) -> bool:
    """Check if a module name matches any exclusion pattern."""
    for pat in patterns:
        if pat in name:
            return True
    return False


def prepare_qat_model(
    model: nn.Module,
    backend: str = 'fbgemm',
    exclude_layers: Optional[List[str]] = None,
    fuse_modules: bool = True,
    inplace: bool = True,
) -> nn.Module:
    """Prepare a model for quantization-aware training (QAT).

    Steps:
      1. Fuse eligible Conv-BN-ReLU / Linear-ReLU patterns.
      2. Assign qconfig to quantizable layers.
      3. Call ``torch.ao.quantization.prepare_qat``.

    Args:
        model: PyTorch module to prepare.
        backend: Quantization backend ('fbgemm' | 'qnnpack' | 'x86').
        exclude_layers: List of layer name substrings to exclude from quantization.
        fuse_modules: Whether to auto-fuse Conv-BN-ReLU patterns before QAT.
        inplace: Whether to modify the model in-place.

    Returns:
        The prepared model (same instance if ``inplace=True``).
    """
    if not inplace:
        model = copy.deepcopy(model)

    # Resolve backend
    if backend in ('fbgemm', 'x86'):
        torch.backends.quantized.engine = 'fbgemm'
    elif backend == 'qnnpack':
        torch.backends.quantized.engine = 'qnnpack'
    else:
        raise ValueError(f"Unsupported quantization backend: {backend}")

    exclude = exclude_layers or ['txt_embed', 'spk_embed', 'lang_embed']

    # Set qconfig
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)

    # Recursively assign qconfig
    def _assign_qconfig(module: nn.Module, prefix: str = ''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if _module_name_matches(full_name, exclude):
                # Keep excluded layers in fp32
                child.qconfig = None
            elif isinstance(child, _QAT_EXCLUDE_TYPES):
                child.qconfig = None
            elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                pass  # keep inherited qconfig
            _assign_qconfig(child, full_name)

    _assign_qconfig(model)

    # Fuse modules
    if fuse_modules:
        model = torch.ao.quantization.fuse_modules(model, inplace=True)

    # Prepare for QAT
    model = torch.ao.quantization.prepare_qat(model.train(), inplace=True)

    return model


def convert_qat_model(model: nn.Module) -> nn.Module:
    """Convert a QAT-trained model to a quantized (int8) model for inference.

    Args:
        model: QAT-prepared model (after training).

    Returns:
        Quantized model with int8 weights and activations.
    """
    model.eval()
    model = torch.ao.quantization.convert(model, inplace=False)
    return model


# ---------------------------------------------------------------------------
# ONNX dtype conversion helpers
# ---------------------------------------------------------------------------

def convert_onnx_to_fp16(onnx_path_or_model, output_path=None, keep_io_types=True):
    """Convert a float32 ONNX model to float16.

    Requires ``onnxconverter-common``.

    Args:
        onnx_path_or_model: Path to .onnx file or onnx.ModelProto.
        output_path: Output path (required if input is a path).
        keep_io_types: Keep input/output tensors as float32.

    Returns:
        onnx.ModelProto in fp16.
    """
    import onnx
    from onnxconverter_common import float16

    if isinstance(onnx_path_or_model, (str,)):
        model = onnx.load(onnx_path_or_model)
    else:
        model = onnx_path_or_model

    model_fp16 = float16.convert_float_to_float16(
        model, keep_io_types=keep_io_types
    )

    if output_path is not None:
        onnx.save(model_fp16, output_path)

    return model_fp16


def validate_onnx_fp16(model, max_nan_nodes: int = 0) -> bool:
    """Check for NaN/Inf after fp16 conversion.

    Returns True if the model is clean.
    """
    import onnx
    import numpy as np

    nan_count = 0
    for initializer in model.graph.initializer:
        if initializer.data_type == 10:  # FLOAT16
            from onnx.numpy_helper import to_array
            arr = to_array(initializer)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                nan_count += 1
    return nan_count <= max_nan_nodes


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

_PRECISION_PROVIDER_MAP = {
    'fp32': ('CUDAExecutionProvider', 'CPUExecutionProvider'),
    'fp16': ('TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'),
    'bf16': ('CUDAExecutionProvider', 'CPUExecutionProvider'),  # ONNX RT doesn't fully support bf16 EP yet
    'int8': ('TensorrtExecutionProvider', 'CPUExecutionProvider'),
}


def get_onnx_providers(precision: str, device: str = 'cuda') -> Tuple[str, ...]:
    """Return recommended ONNX Runtime execution providers for a given precision."""
    if device == 'cpu':
        return ('CPUExecutionProvider',)
    return _PRECISION_PROVIDER_MAP.get(precision, ('CPUExecutionProvider',))


# ---------------------------------------------------------------------------
# CLI helpers for common argument parsing
# ---------------------------------------------------------------------------

def add_precision_arguments(parser_or_group):
    """Add common precision-related CLI arguments to an argparse parser/group."""
    parser_or_group.add_argument(
        '--precision',
        type=str,
        default='fp32',
        choices=sorted(VALID_EXPORT_PRECISIONS),
        help='Target precision for export/inference (default: fp32). '
             'Use "all" to export all supported precisions.',
    )
    parser_or_group.add_argument(
        '--quantize',
        action='store_true',
        default=False,
        help='Additionally export int8 quantized ONNX model.',
    )
