import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticDataset
from utils.hparams import set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def main():
    parser = argparse.ArgumentParser(description='Measure reflow prediction loss at fixed times.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--split', choices=('train', 'valid'), default='valid')
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates the full split')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')

    set_hparams(config=args.config, print_hparams=False)
    torch.set_num_threads(1)
    dataset = AcousticDataset(args.split)
    model = DiffSingerAcoustic(len(load_phoneme_dictionary()), 128).cuda().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cuda', weights_only=True))
    times = (0.4, 0.55, 0.7, 0.85, 0.99)
    sums = [0.0] * len(times)
    elements = 0
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    with torch.inference_mode():
        for index in range(count):
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
            elements += target.numel()
            for i, time in enumerate(times):
                torch.manual_seed(42 + index)
                step = torch.tensor([time], device='cuda')
                prediction, truth = model.diffusion.p_losses(target, step, condition.transpose(1, 2))
                sums[i] += (prediction - truth).square().sum().item()
    print('TIMESTEP_LOSS', args.split, 'items', count, dict(zip(times, (x / elements for x in sums))))


if __name__ == '__main__':
    main()
