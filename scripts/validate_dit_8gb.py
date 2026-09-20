import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.toplevel import DiffSingerAcoustic
from scripts.validate_dit_configs import load_config
from utils.hparams import hparams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=768)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--accumulate', type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the 8GB DiT training smoke test.')

    hparams.clear()
    hparams.update(load_config(ROOT / 'configs/templates/config_acoustic_dit.yaml'))
    assert args.frames <= hparams['max_sample_frames']

    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler_cls = getattr(torch.amp, 'GradScaler')
    scaler = scaler_cls('cuda', enabled=dtype == torch.float16)
    model = DiffSingerAcoustic(vocab_size=64, out_dims=hparams['audio_num_mel_bins']).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=0.01
    )

    token_count = 32
    tokens = torch.randint(1, 64, (1, token_count), device=device)
    frame_indices = torch.arange(args.frames, device=device)
    mel2ph = (frame_indices * token_count // args.frames + 1)[None]
    f0 = torch.full((1, args.frames), 220.0, device=device)
    target = torch.randn(1, args.frames, hparams['audio_num_mel_bins'], device=device)
    valid = (mel2ph > 0).unsqueeze(1).unsqueeze(1)

    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    for _ in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        for _ in range(args.accumulate):
            with torch.autocast('cuda', dtype=dtype):
                output = model(
                    tokens,
                    mel2ph=mel2ph,
                    f0=f0,
                    key_shift=torch.zeros(1, 1, device=device),
                    speed=torch.ones(1, 1, device=device),
                    gt_mel=target,
                    infer=False,
                )
                velocity, velocity_target, _ = output.diff_out
                diffusion_loss = ((velocity - velocity_target).square() * valid).sum()
                diffusion_loss = diffusion_loss / valid.expand_as(velocity).sum()
                auxiliary_target = model.aux_decoder.norm_spec(target)
                auxiliary_loss = F.mse_loss(output.aux_out, auxiliary_target)
                loss = (diffusion_loss + auxiliary_loss) / args.accumulate
            scaler.scale(loss).backward()
            accumulated_loss += loss.detach()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['clip_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(accumulated_loss):
            raise AssertionError('DiT acoustic training produced a non-finite loss.')
        losses.append(float(accumulated_loss))

    allocated = torch.cuda.max_memory_allocated(device) / 2 ** 20
    reserved = torch.cuda.max_memory_reserved(device) / 2 ** 20
    device_name = torch.cuda.get_device_name(device)
    print(
        f'DiT 8GB training smoke passed: device={device_name}, dtype={dtype}, '
        f'frames={args.frames}, updates={args.updates}, accumulate={args.accumulate}, '
        f'losses={losses}, max_allocated={allocated:.1f} MiB, '
        f'max_reserved={reserved:.1f} MiB.'
    )


if __name__ == '__main__':
    main()