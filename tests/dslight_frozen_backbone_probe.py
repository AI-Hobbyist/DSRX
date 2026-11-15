import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()))
from modules.backbones import build_backbone
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticDataset
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def cache_conditions(model, split):
    dataset = AcousticDataset(split)
    result = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset.collater([dataset[index]])
            sample = {
                key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }
            condition = model.fs2(
                sample['tokens'], sample['mel2ph'], sample['f0'],
                key_shift=sample['key_shift'], speed=sample['speed'],
            )
            target = model.diffusion.norm_spec(sample['mel']).transpose(-2, -1)[:, None]
            result.append((condition.transpose(1, 2), target))
    return result


def validate(predictor, samples):
    squared_error = 0.0
    elements = 0
    with torch.no_grad():
        for index, (condition, target) in enumerate(samples):
            torch.manual_seed(42 + index)
            noise = torch.randn_like(target)
            state = noise + 0.55 * (target - noise)
            step = torch.tensor([550.0], device='cuda')
            prediction = predictor(state, step, condition)
            squared_error += (prediction - (target - noise)).square().sum().item()
            elements += target.numel()
    return squared_error / elements


def main():
    parser = argparse.ArgumentParser(description='Train a backbone with a frozen baseline encoder.')
    parser.add_argument('--config', default='data/dslight_eval100/config.yaml')
    parser.add_argument('--weights', default='data/dslight_eval100/legacy_3000_steps.pth')
    parser.add_argument('--steps', type=int, default=1500)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')

    set_hparams(config=args.config, print_hparams=False)
    legacy = set_hparams(config='configs/original/acoustic.yaml', print_hparams=False, global_hparams=False)
    hparams['backbone_type'] = legacy['backbone_type']
    hparams['backbone_args'] = legacy['backbone_args']
    hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
    hparams['shallow_diffusion_args']['aux_decoder_args'] = legacy['shallow_diffusion_args']['aux_decoder_args']
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = DiffSingerAcoustic(len(load_phoneme_dictionary()), 128).cuda().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cuda', weights_only=True))
    train = cache_conditions(model, 'train')
    valid = cache_conditions(model, 'valid')

    torch.manual_seed(123)
    backbone = build_backbone(128, 1, 'dslight', {}).cuda().train()
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=0.0006)
    for step_index in range(args.steps):
        condition, target = train[step_index % len(train)]
        step = 0.4 + 0.6 * torch.rand((1,), device='cuda')
        noise = torch.randn_like(target)
        state = noise + step[:, None, None, None] * (target - noise)
        optimizer.zero_grad(set_to_none=True)
        prediction = backbone(state, step * 1000, condition)
        loss = nn.functional.mse_loss(prediction, target - noise)
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at step {step_index}')
        loss.backward()
        optimizer.step()

    backbone.eval()
    print('FROZEN_BACKBONE legacy t=0.55 l2', validate(model.diffusion.velocity_fn, valid))
    print('FROZEN_BACKBONE dslight t=0.55 l2', validate(backbone, valid))


if __name__ == '__main__':
    main()
