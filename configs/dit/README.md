# DiT configuration, training and deployment

This directory contains the frozen P0 configuration contract, P1 floating-point
training templates, P2 native PyTorch inference policy and P3 floating-point
ONNX export contract. Quantization, distillation and GGUF remain later phases.

## Files

- `acoustic.yaml` and `variance.yaml` are architecture-specific cascade layers.
- `config_acoustic.yaml` and `config_variance.yaml` are user-editable templates.
- `*_4090_10h.yaml` files provide recommended starting overlays and editable
  templates for one RTX 4090 with about 10 hours of aligned data.
- `_reset_*.yaml` clears inherited convolution-only arguments before DiT
  arguments are merged.
- `p0_baseline.yaml` records the repository baseline, tensor/checkpoint
  contracts and the minimum compatibility matrix.

No acoustic or variance checkpoint is present in this workspace. The recorded
validation therefore covers construction, tensor/mask behavior, gradients,
synthetic training, native inference and legacy call compatibility, but does
not claim numerical regression, convergence or audio quality against trained
weights.

The reset layers are required because `utils.hparams.override_config()` merges
nested mappings key by key. Without the reset, fields such as `kernel_size` and
`dropout_rate` would survive from the LYNXNet2 configuration and could be
silently discarded by the current `filter_kwargs()` path.

Run the configuration and P1 behavior validations with:

```powershell
D:/UserData/Conda/envs/diffs/python.exe scripts/validate_dit_configs.py
D:/UserData/Conda/envs/diffs/python.exe scripts/validate_dit_p1.py
D:/UserData/Conda/envs/diffs/python.exe scripts/validate_dit_p2.py --gpu
D:/UserData/Conda/envs/diffs/python.exe scripts/validate_dit_p3.py --runtime
```

On an approximately 8 GiB CUDA device, the bounded training smoke is:

```powershell
D:/UserData/Conda/envs/diffs/python.exe scripts/validate_dit_8gb.py --updates 20
```

The 2026-09-19 reference run used an RTX 5060 Laptop GPU (8151 MiB), BF16,
768 frames, micro-batch 1 and accumulation 8. It completed 20 optimizer updates
with finite losses and measured 835.8 MiB allocated / 890.0 MiB reserved peak
PyTorch memory. This is a synthetic stability check, not a convergence or audio
quality result.

P2 keeps mask-aware DiT backbones eager so every call supplies its dynamic
padding mask. Legacy three-argument backbones retain the TorchScript path.
Only plain `nn.Linear` modules may use selective FP16; LoRA and other Linear
subclasses are preserved. DiT warmup defaults to 128/512/768/1024 frames, and
native inference rejects requests beyond `inference_max_frames: 2048` instead
of silently cropping aligned inputs.

The 2026-09-19 P2 CUDA run on the same RTX 5060 validated DDPM/Reflow,
acoustic B=2 padding, pitch plus three multi-variance outputs, legacy LYNXNet2,
model-switch rollback and strict inference LoRA loading. A 768-frame 20-step
Reflow sample took 0.385 seconds with 67.8 MiB peak allocated PyTorch memory;
a single 2048-frame backbone boundary forward took 0.025 seconds with 90.0 MiB
peak allocated memory. These are synthetic engineering measurements, not
trained-model quality or end-to-end audio latency claims.

P3 keeps the existing batch-1 host protocol and opset 15 package shape. A
deployment-only DiT adapter creates an all-valid mask from the dynamic condition
length and lowers attention to MatMul, Softmax and RoPE basic operators. The
2026-09-19 validation used PyTorch 2.11's legacy TorchScript exporter, ONNX
1.16.2 and ONNX Runtime 1.23.0 CPU execution. Checker and numerical comparisons
passed at 1, 7 and 16 frames, as did the legacy three-argument backbone and the
shallow Reflow wrapper. No local PyTorch 1.13 environment or trained acoustic /
variance checkpoint was available, so the exact locked exporter environment,
finished model package, host integration and audio quality are not claimed as
measured.

## RTX 4090 / approximately 10-hour starting profile

These values are recommendations, not measurements on an RTX 4090. They keep
the 22,019,456-parameter acoustic DiT and both variance architectures unchanged:

| Model | Micro-batch (`max_batch_size`) | `max_batch_frames` | Validation interval |
| --- | ---: | ---: | ---: |
| Acoustic | 8 | 6144 | 500 trainer batches |
| Variance | 16 | 12288 | 500 trainer batches |

Both profiles use `max_sample_frames: 768`, `accumulate_grad_batches: 1`,
validation batch 1 and `bf16-mixed`. `max_batch_frames` is a total frame budget,
so variable-length batches may contain fewer samples than the micro-batch cap.
Start with [config_acoustic_4090_10h.yaml](config_acoustic_4090_10h.yaml) or
[config_variance_4090_10h.yaml](config_variance_4090_10h.yaml), then reduce the
batch cap and frame budget together if the complete training or validation path
exceeds dedicated VRAM.

## Later phase prerequisites

P4-P6 require trained model artifacts and cannot be completed from random
weights alone. P4 needs a trained floating-point DiT plus representative
calibration/evaluation data; P5 needs a trained, quality-qualified multi-step
teacher; P6 needs a trained floating-point DiT for graph, numerical and
quantized execution comparisons.

`max_sample_frames: 768` is enforced before dataset construction. Oversized
items must be split at aligned phrase, word or note boundaries before
binarization; the loader does not perform unsafe frame-only cropping.