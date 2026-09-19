import sys
import argparse
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import optimization
from inference import batch_backend
from modules.backbones.dit import DiT
from modules.core.ddpm import GaussianDiffusion
from modules.core.reflow import RectifiedFlow
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from scripts.validate_dit_configs import load_config
from utils.hparams import hparams
from utils.lora import LoRALinear, load_lora_state_dict


def expect_error(error_type, fn, message: str) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(message)


def validate_linear_protection() -> None:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.normal = nn.Linear(128, 128)
            self.adapter = LoRALinear(128, 128, r=8, alpha=16)

    model = Model()
    converted, _ = optimization._replace_large_linears(model, 1)
    assert converted == 1
    assert isinstance(model.normal, optimization.SelectiveFP16Linear)
    assert isinstance(model.adapter, LoRALinear)

    state = {
        'model.adapter.lora_A': torch.ones_like(model.adapter.lora_A),
        'model.adapter.lora_B': torch.ones_like(model.adapter.lora_B),
    }
    loaded = load_lora_state_dict(model, state, strict=True)
    assert loaded == ['model.adapter.lora_A', 'model.adapter.lora_B']
    expect_error(
        KeyError,
        lambda: load_lora_state_dict(
            model, {'model.adapter.lora_A': state['model.adapter.lora_A']},
            strict=True,
        ),
        'Strict DiT LoRA loading must reject incomplete adapter checkpoints.',
    )


def validate_dit_inference_policy() -> None:
    hparams.clear()
    hparams.update(load_config(ROOT / 'configs/dit/config_acoustic.yaml'))
    hparams['hidden_size'] = 256
    model = DiT(
        128,
        1,
        num_layers=1,
        num_channels=64,
        num_heads=1,
        mlp_ratio=2,
        time_embed_dim=64,
        use_gradient_checkpointing=False,
    )

    warmup_policy = getattr(optimization, 'get_backbone_warmup_frames', None)
    assert callable(warmup_policy), 'Missing backend-aware warmup policy.'
    assert tuple(warmup_policy(model)) == (128, 512, 768, 1024)

    length_validator = getattr(optimization, 'validate_inference_length', None)
    assert callable(length_validator), 'Missing inference length validation.'
    length_validator(2048, context='acoustic')
    expect_error(
        ValueError,
        lambda: length_validator(2049, context='acoustic'),
        'DiT inference must reject requests beyond the configured frame limit.',
    )


def validate_backbone_discovery() -> None:
    class Predictor(nn.Module):
        def __init__(self, attr):
            super().__init__()
            setattr(self, attr, nn.Identity())

    model = nn.Module()
    model.reflow = Predictor('velocity_fn')
    model.ddpm = Predictor('denoise_fn')
    discover = getattr(optimization, '_prediction_backbones', None)
    assert callable(discover), 'Missing DDPM/Reflow backbone discovery.'
    targets = list(discover(model))
    assert {target.attribute for target in targets} == {'velocity_fn', 'denoise_fn'}


def validate_variance_model_switch_rollback() -> None:
    backend = batch_backend.BatchInferenceBackend.__new__(
        batch_backend.BatchInferenceBackend
    )
    backend.device = 'cpu'
    backend.ckpt_steps = None
    backend._variance_infer = object()
    backend._variance_predictions = {'pitch'}
    backend._last_active = 0.0
    releases = []
    backend._release_model_resources = lambda: releases.append(True)
    previous = backend._variance_infer
    original = batch_backend.DiffSingerVarianceInfer

    class Failure:
        def __init__(self, **_kwargs):
            raise RuntimeError('expected load failure')

    try:
        batch_backend.DiffSingerVarianceInfer = Failure
        expect_error(
            RuntimeError,
            lambda: backend._load_variance_model(lambda *_args: None, {'energy'}),
            'Variance model replacement failure must propagate.',
        )
        assert backend._variance_infer is previous
        assert backend._variance_predictions == {'pitch'}
        assert releases == [True]
    finally:
        batch_backend.DiffSingerVarianceInfer = original


def validate_native_sampling() -> None:
    hparams.clear()
    hparams.update({
        'hidden_size': 8,
        'schedule_type': 'linear',
        'use_shallow_diffusion': False,
        'sampling_algorithm': 'euler',
        'sampling_steps': 2,
        'diff_speedup': 1,
        'infer': False,
    })
    backbone_args = {
        'num_layers': 1,
        'num_channels': 8,
        'num_heads': 2,
        'mlp_ratio': 2,
        'time_embed_dim': 8,
        'use_gradient_checkpointing': False,
    }
    ddpm = GaussianDiffusion(
        4,
        timesteps=2,
        k_step=2,
        backbone_type='dit',
        backbone_args=backbone_args,
        spec_min=[-1.0],
        spec_max=[1.0],
    ).eval()
    reflow = RectifiedFlow(
        4,
        time_scale_factor=1000,
        backbone_type='dit',
        backbone_args=backbone_args,
        spec_min=[-1.0],
        spec_max=[1.0],
    ).eval()

    for frames in (1, 9, 3):
        condition = torch.randn(1, frames, 8)
        mask = torch.ones(1, frames, dtype=torch.bool)
        for predictor in (ddpm, reflow):
            output = predictor(condition, infer=True, valid_mask=mask)
            assert output.shape == (1, frames, 4)
            assert torch.isfinite(output).all()

    condition = torch.randn(2, 7, 8)
    mask = torch.tensor([[True, True, True, False, False, False, False], [True] * 7])
    for predictor in (ddpm, reflow):
        output = predictor(condition, infer=True, valid_mask=mask)
        assert output.shape == (2, 7, 4)
        assert torch.isfinite(output).all()


def validate_cuda_optimization() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA validation requested but CUDA is unavailable.')
    device = torch.device('cuda')
    hparams.clear()
    hparams.update({
        'hidden_size': 256,
        'inference_optimization': True,
        'inference_selective_fp16': True,
        'inference_selective_fp16_min_elements': 16_384,
        'inference_torchscript': True,
        'inference_dit_warmup_frames': [128, 512, 768, 1024],
        'sampling_algorithm': 'euler',
        'sampling_steps': 20,
        'use_shallow_diffusion': False,
        'infer': False,
    })
    reflow = RectifiedFlow(
        128,
        time_scale_factor=1000,
        backbone_type='dit',
        backbone_args={
            'num_layers': 8,
            'num_channels': 384,
            'num_heads': 6,
            'mlp_ratio': 4,
            'time_embed_dim': 256,
            'use_gradient_checkpointing': False,
        },
        spec_min=[-1.0],
        spec_max=[1.0],
    ).eval().to(device)
    report = optimization.optimize_model_for_inference(
        reflow, model_kind='acoustic', device=device
    )
    assert len(report.backbones) == 1
    assert report.backbones[0].backend == 'eager_masked'
    assert report.backbones[0].warmup_frames == [128, 512, 768, 1024]

    torch.cuda.reset_peak_memory_stats(device)
    condition = torch.randn(1, 768, 256, device=device)
    valid_mask = torch.ones(1, 768, dtype=torch.bool, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = reflow(condition, infer=True, valid_mask=valid_mask)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    assert output.shape == (1, 768, 128)
    assert torch.isfinite(output).all()

    torch.cuda.reset_peak_memory_stats(device)
    boundary_spec = torch.randn(1, 1, 128, 2048, device=device)
    boundary_cond = torch.randn(1, 256, 2048, device=device)
    boundary_mask = torch.ones(1, 2048, dtype=torch.bool, device=device)
    torch.cuda.synchronize(device)
    boundary_started = time.perf_counter()
    boundary_output = reflow.velocity_fn(
        boundary_spec,
        torch.tensor([500.0], device=device),
        boundary_cond,
        valid_mask=boundary_mask,
    )
    torch.cuda.synchronize(device)
    boundary_elapsed = time.perf_counter() - boundary_started
    boundary_peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    assert boundary_output.shape == boundary_spec.shape
    assert torch.isfinite(boundary_output).all()

    class LegacyBackbone(nn.Module):
        def forward(self, spec, diffusion_step, cond):
            return spec + diffusion_step[:, None, None, None] * 0 + cond[:, None, :1, :] * 0

    class LegacyOwner(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_feats = 1
            self.out_dims = 4
            self.velocity_fn = LegacyBackbone()

    hparams['hidden_size'] = 8
    hparams['inference_selective_fp16'] = False
    hparams['inference_warmup_frames'] = [8, 12]
    legacy = LegacyOwner().eval().to(device)
    legacy_report = optimization.optimize_model_for_inference(
        legacy, model_kind='variance', device=device
    )
    assert legacy_report.torchscript
    assert legacy_report.backbones[0].backend == 'torchscript'
    print(
        'DiT P2 CUDA metrics: '
        f'frames=768, steps=20, seconds={elapsed:.3f}, '
        f'peak_allocated_mib={peak_mib:.1f}, '
        f'converted_linears={report.converted_linears}, '
        f'boundary_frames=2048, boundary_seconds={boundary_elapsed:.3f}, '
        f'boundary_peak_allocated_mib={boundary_peak_mib:.1f}'
    )
    del (
        reflow, legacy, output, condition, valid_mask,
        boundary_spec, boundary_cond, boundary_mask, boundary_output,
    )
    torch.cuda.empty_cache()


def validate_cuda_top_level_inference() -> None:
    device = torch.device('cuda')
    hparams.clear()
    hparams.update(load_config(ROOT / 'configs/dit/config_acoustic.yaml'))
    hparams.update({
        'sampling_steps': 2,
        'use_spk_id': False,
        'use_lang_id': False,
        'infer': False,
    })
    acoustic = DiffSingerAcoustic(
        vocab_size=64,
        out_dims=hparams['audio_num_mel_bins'],
    ).eval().to(device)
    frames = 96
    tokens = torch.randint(1, 64, (2, 8), device=device)
    mel2ph = torch.stack((
        torch.cat((torch.arange(1, 58, device=device) % 8 + 1, torch.zeros(39, device=device))),
        torch.arange(frames, device=device) % 8 + 1,
    )).long()
    f0 = torch.full((2, frames), 220.0, device=device)
    acoustic_output = acoustic(
        tokens,
        mel2ph=mel2ph,
        f0=f0,
        key_shift=torch.zeros(2, 1, device=device),
        speed=torch.ones(2, 1, device=device),
        infer=True,
    ).diff_out
    assert acoustic_output.shape == (2, frames, hparams['audio_num_mel_bins'])
    assert torch.count_nonzero(acoustic_output[0, 57:]) == 0
    assert torch.isfinite(acoustic_output).all()
    del acoustic, acoustic_output
    torch.cuda.empty_cache()

    hparams.clear()
    hparams.update(load_config(ROOT / 'configs/dit/config_variance.yaml'))
    hparams.update({
        'sampling_steps': 2,
        'predict_pitch': True,
        'predict_breathiness': True,
        'predict_voicing': True,
        'predict_tension': True,
        'use_spk_id': False,
        'use_lang_id': False,
        'infer': False,
    })
    variance = DiffSingerVariance(vocab_size=64).eval().to(device)
    frames = 64
    tokens = torch.randint(1, 64, (2, 4), device=device)
    ph2word = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2]], device=device)
    ph_dur = torch.tensor([[8, 8, 8, 8], [16, 16, 16, 16]], device=device)
    word_dur = torch.tensor([[16, 16], [32, 32]], device=device)
    mel2ph = torch.stack((
        torch.cat((torch.arange(32, device=device) // 8 + 1, torch.zeros(32, device=device))),
        torch.arange(frames, device=device) // 16 + 1,
    )).long()
    base_pitch = torch.full((2, frames), 60.0, device=device)
    _, pitch_output, variance_outputs = variance(
        tokens,
        midi=torch.full((2, 4), 60, dtype=torch.long, device=device),
        ph2word=ph2word,
        ph_dur=ph_dur,
        word_dur=word_dur,
        mel2ph=mel2ph,
        base_pitch=base_pitch,
        infer=True,
    )
    assert pitch_output.shape == (2, frames)
    assert torch.isfinite(pitch_output).all()
    assert variance_outputs
    for output in variance_outputs.values():
        assert output.shape == (2, frames)
        assert torch.isfinite(output).all()
    print(
        'DiT P2 top-level inference passed: '
        f'acoustic_batch=2x96, variance_batch=2x64, '
        f'variance_outputs={sorted(variance_outputs)}'
    )


def validate_cuda_legacy_inference() -> None:
    device = torch.device('cuda')
    hparams.clear()
    hparams.update(load_config(ROOT / 'configs/original/acoustic.yaml'))
    hparams.update({
        'sampling_steps': 2,
        'use_spk_id': False,
        'use_lang_id': False,
        'infer': False,
    })
    model = DiffSingerAcoustic(
        vocab_size=64,
        out_dims=hparams['audio_num_mel_bins'],
    ).eval().to(device)
    frames = 32
    output = model(
        torch.randint(1, 64, (1, 4), device=device),
        mel2ph=(torch.arange(frames, device=device) % 4 + 1)[None],
        f0=torch.full((1, frames), 220.0, device=device),
        key_shift=torch.zeros(1, 1, device=device),
        speed=torch.ones(1, 1, device=device),
        infer=True,
    ).diff_out
    assert output.shape == (1, frames, hparams['audio_num_mel_bins'])
    assert torch.isfinite(output).all()
    print('Legacy acoustic inference passed: backbone=lynxnet2, batch=1x32')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(1234)
    validate_linear_protection()
    validate_dit_inference_policy()
    validate_backbone_discovery()
    validate_variance_model_switch_rollback()
    validate_native_sampling()
    if args.gpu:
        validate_cuda_optimization()
        validate_cuda_top_level_inference()
        validate_cuda_legacy_inference()
    print('DiT P2 validation passed.')


if __name__ == '__main__':
    main()