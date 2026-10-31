import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()))
from modules.aux_decoder import AuxDecoderAdaptor
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticDataset
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def cache_conditions(model, prefix, device):
    dataset = AcousticDataset(prefix)
    result = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset.collater([dataset[index]])
            sample = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }
            condition = model.fs2(
                sample['tokens'], sample['mel2ph'], sample['f0'],
                key_shift=sample['key_shift'], speed=sample['speed'],
            )
            result.append((condition.detach(), sample['mel']))
    return result


def compare_decoder(name, decoder_args, train, valid, steps, device):
    torch.manual_seed(123)
    decoder = AuxDecoderAdaptor(256, 128, 1, [-12], [0], name, decoder_args).to(device).train()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.0001)
    for step in range(steps):
        condition, mel = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.l1_loss(decoder(condition), decoder.norm_spec(mel))
        if not torch.isfinite(loss):
            raise RuntimeError(f'{name} non-finite at step {step}')
        loss.backward()
        optimizer.step()

    decoder.eval()
    error = 0.0
    count = 0
    with torch.no_grad():
        for condition, mel in valid:
            error += (decoder(condition, infer=True) - mel).abs().sum().item()
            count += mel.numel()
    print('FROZEN_ENCODER', name, 'steps', steps, 'valid_l1', error / count, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Compare auxiliary decoders with one frozen encoder.')
    parser.add_argument('--config', default='data/dslight_eval100/config.yaml')
    parser.add_argument('--weights', default='data/dslight_eval100/legacy_3000_steps.pth')
    parser.add_argument('--steps', type=int, default=1500)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')

    set_hparams(config=args.config, print_hparams=False)
    old = set_hparams(config='configs/original/acoustic.yaml', print_hparams=False, global_hparams=False)
    hparams['backbone_type'] = old['backbone_type']
    hparams['backbone_args'] = old['backbone_args']
    hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
    hparams['shallow_diffusion_args']['aux_decoder_args'] = old['shallow_diffusion_args']['aux_decoder_args']
    torch.set_num_threads(1)
    torch.manual_seed(123)
    device = torch.device('cuda')
    model = DiffSingerAcoustic(len(load_phoneme_dictionary()), 128).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    train = cache_conditions(model, 'train', device)
    valid = cache_conditions(model, 'valid', device)
    dslight_args = {
        'num_channels': 256,
        'num_layers': 8,
        'kernel_size': 7,
        'dilation_cycle': [1, 3, 9],
        'dropout_rate': 0.0,
    }
    compare_decoder('convnext', old['shallow_diffusion_args']['aux_decoder_args'], train, valid, args.steps, device)
    compare_decoder('dslight', dslight_args, train, valid, args.steps, device)


if __name__ == '__main__':
    main()
