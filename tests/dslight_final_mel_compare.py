import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticDataset
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def evaluate(name, weights, config, device, limit):
    set_hparams(config=config, print_hparams=False)
    if name == 'legacy':
        old = set_hparams(config='configs/original/acoustic.yaml', print_hparams=False, global_hparams=False)
        hparams['backbone_type'] = old['backbone_type']
        hparams['backbone_args'] = old['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = old['shallow_diffusion_args']['aux_decoder_args']

    dataset = AcousticDataset('valid')
    model = DiffSingerAcoustic(len(load_phoneme_dictionary()), 128).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    total_error = 0.0
    total_elements = 0
    count = min(limit, len(dataset)) if limit else len(dataset)
    with torch.inference_mode():
        for index in range(count):
            sample = dataset.collater([dataset[index]])
            sample = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }
            torch.manual_seed(42 + index)
            result = model(
                sample['tokens'], mel2ph=sample['mel2ph'], f0=sample['f0'],
                key_shift=sample['key_shift'], speed=sample['speed'], infer=True,
            )
            if not torch.isfinite(result.diff_out).all():
                raise RuntimeError(f'Non-finite inference output in {name} item {index}')
            total_error += (result.diff_out - sample['mel']).abs().sum().item()
            total_elements += sample['mel'].numel()
    print('FINAL_MEL', name, 'items', count, 'steps', hparams['sampling_steps'],
          'l1', total_error / total_elements, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Compare sampled mel outputs against validation targets.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--legacy-config', help='Baseline config when encoder settings differ')
    parser.add_argument('--legacy-weights', required=True)
    parser.add_argument('--dslight-weights', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates all validation items')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')
    torch.set_num_threads(1)
    for name, weights in (('legacy', args.legacy_weights), ('dslight', args.dslight_weights)):
        config = args.legacy_config if name == 'legacy' and args.legacy_config else args.config
        evaluate(name, weights, config, args.device, args.limit)


if __name__ == '__main__':
    main()
