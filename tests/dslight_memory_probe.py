import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from modules.toplevel import DiffSingerAcoustic, DiffSingerAllInOne, DiffSingerVariance
from utils.hparams import set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def acoustic_loss(model, tokens, alignment, curve):
    frames = alignment.shape[1]
    output = model(
        tokens, mel2ph=alignment, f0=torch.full_like(curve, 220),
        key_shift=torch.zeros(1, 1, device='cuda'), speed=torch.ones(1, 1, device='cuda'),
        energy=curve, breathiness=curve, voicing=curve, tension=curve,
        gt_mel=torch.full((1, frames, 128), -6.0, device='cuda'), infer=False,
    )
    return output.aux_out.square().mean() + output.diff_out[0].square().mean()


def variance_loss(model, tokens, alignment, curve):
    phonemes = tokens.shape[1]
    duration, pitch, variance = model(
        tokens, midi=torch.full_like(tokens, 60), ph2word=(torch.arange(phonemes, device='cuda')[None] // 2 + 1),
        ph_dur=torch.full_like(tokens, alignment.shape[1] // phonemes), mel2ph=alignment,
        base_pitch=torch.full_like(curve, 60), pitch=torch.full_like(curve, 60),
        energy=curve, breathiness=curve, voicing=curve, tension=curve, infer=False,
    )
    return duration.square().mean() + pitch[0].square().mean() + variance[0].square().mean()


def measure(config, kind, frames, phonemes):
    set_hparams(config=config, print_hparams=False)
    torch.manual_seed(123)
    vocab = len(load_phoneme_dictionary())
    if kind == 'acoustic':
        model = DiffSingerAcoustic(vocab, 128)
    elif kind == 'variance':
        model = DiffSingerVariance(vocab)
    else:
        model = DiffSingerAllInOne(vocab, 128)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    tokens = torch.ones((1, phonemes), dtype=torch.long, device='cuda')
    alignment = (torch.arange(frames, device='cuda')[None] // (frames // phonemes) + 1).long()
    curve = torch.zeros((1, frames), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        if kind == 'acoustic':
            loss = acoustic_loss(model, tokens, alignment, curve)
        elif kind == 'variance':
            loss = variance_loss(model, tokens, alignment, curve)
        else:
            loss = acoustic_loss(model.acoustic, tokens, alignment, curve)
            loss = loss + variance_loss(model.variance, tokens, alignment, curve)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite synthetic loss')
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    result = {'config': config, 'kind': kind, 'frames_per_stream': frames, 'phonemes': phonemes,
              'batch_per_stream': 1, 'precision': 'fp32', 'optimizer': 'AdamW',
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    print('SYNTHETIC_MEMORY', json.dumps(result), flush=True)
    del model, optimizer, loss, tokens, alignment, curve
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Measure synthetic FP32 training memory.')
    parser.add_argument('--frames', type=int, default=512)
    parser.add_argument('--phonemes', type=int, default=64)
    args = parser.parse_args()
    if args.phonemes < 1 or args.frames < args.phonemes or args.frames % args.phonemes:
        parser.error('--frames must be a positive multiple of --phonemes')
    torch.set_num_threads(1)
    configs = (
        'configs/templates/all_in_one_dslight_50mb.yaml',
        'configs/templates/all_in_one_dslight_quality_50mb.yaml',
    )
    for config in configs:
        for kind in ('acoustic', 'variance', 'all_in_one'):
            measure(config, kind, args.frames, args.phonemes)


if __name__ == '__main__':
    main()
