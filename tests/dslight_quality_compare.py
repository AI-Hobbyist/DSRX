import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()))
from modules.losses import RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticDataset, AcousticTask
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def evaluate(name, weights, config, channels, layers, limit):
    set_hparams(config=config, print_hparams=False)
    if name == 'legacy':
        legacy = set_hparams(
            config='configs/original/acoustic.yaml', print_hparams=False, global_hparams=False,
        )
        hparams['backbone_type'] = legacy['backbone_type']
        hparams['backbone_args'] = legacy['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = legacy['shallow_diffusion_args']['aux_decoder_args']
    else:
        decoder_args = hparams['shallow_diffusion_args']['aux_decoder_args']
        decoder_args['num_channels'] = channels
        decoder_args['num_layers'] = layers

    dataset = AcousticDataset('valid')
    model = DiffSingerAcoustic(len(load_phoneme_dictionary()), 128).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    task = SimpleNamespace(
        model=model,
        required_variances=[],
        diffusion_type='reflow',
        lambda_aux_mel_loss=hparams['lambda_aux_mel_loss'],
        aux_mel_loss=nn.L1Loss(),
        mel_loss=RectifiedFlowLoss(loss_type='l2', log_norm=False),
    )
    error = 0.0
    elements = 0
    main_loss = 0.0
    frames = 0
    count = min(limit, len(dataset)) if limit else len(dataset)
    with torch.no_grad():
        for index in range(count):
            sample = dataset.collater([dataset[index]])
            condition = model.fs2(
                sample['tokens'], sample['mel2ph'], sample['f0'],
                key_shift=sample['key_shift'], speed=sample['speed'],
            )
            prediction = model.aux_decoder(condition, infer=True)
            error += (prediction - sample['mel']).abs().sum().item()
            elements += sample['mel'].numel()
            torch.manual_seed(index + 42)
            losses = AcousticTask.run_model(task, sample)
            nframes = int(sample['mel2ph'].gt(0).sum())
            main_loss += losses['mel_loss'].item() * nframes
            frames += nframes
    print('COMPARE', name, 'items', count, 'aux_l1', error / elements, 'main_l2', main_loss / frames)


def main():
    parser = argparse.ArgumentParser(description='Compare a baseline and DSLight acoustic checkpoint.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--legacy-weights', required=True)
    parser.add_argument('--dslight-weights', required=True)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates all validation items')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')
    torch.set_num_threads(1)
    for name, weights in (('legacy', args.legacy_weights), ('dslight', args.dslight_weights)):
        evaluate(name, weights, args.config, args.channels, args.layers, args.limit)


if __name__ == '__main__':
    main()
