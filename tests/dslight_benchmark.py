import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.cwd()))
from modules.losses import RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticDataset, AcousticTask
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary

parser = argparse.ArgumentParser(description='Benchmark acoustic training on a prepared dataset.')
parser.add_argument('model', choices=('legacy', 'dslight'))
parser.add_argument('config', nargs='?', default='data/dslight_smoke/config.yaml')
parser.add_argument('--steps', type=int, default=60)
parser.add_argument('--aux-decoder-grad', type=float)
parser.add_argument('--save-weights', type=Path)
args = parser.parse_args()
if args.steps <= 10:
    parser.error('--steps must exceed the 10-step warmup')

set_hparams(config=args.config, print_hparams=False)
if args.model == 'legacy':
    old = set_hparams(config='configs/original/acoustic.yaml', print_hparams=False, global_hparams=False)
    hparams['backbone_type'] = old['backbone_type']
    hparams['backbone_args'] = old['backbone_args']
    hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
    hparams['shallow_diffusion_args']['aux_decoder_args'] = old['shallow_diffusion_args']['aux_decoder_args']

if args.aux_decoder_grad is not None:
    hparams['shallow_diffusion_args']['aux_decoder_grad'] = args.aux_decoder_grad

torch.manual_seed(123)
torch.set_num_threads(1)
device = torch.device('cuda')
dataset = AcousticDataset('train')
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collater)
model = DiffSingerAcoustic(len(load_phoneme_dictionary()), 128).to(device).train()
task = SimpleNamespace(
    model=model, required_variances=[], diffusion_type='reflow',
    lambda_aux_mel_loss=hparams['lambda_aux_mel_loss'], aux_mel_loss=nn.L1Loss(),
    mel_loss=RectifiedFlowLoss(loss_type=hparams['main_loss_type'], log_norm=hparams['main_loss_log_norm']),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
torch.cuda.reset_peak_memory_stats()
times = []
frames = []
iterator = iter(loader)
for step in range(args.steps):
    started = time.perf_counter()
    try:
        sample = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        sample = next(iterator)
    sample = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    optimizer.zero_grad(set_to_none=True)
    loss = sum(AcousticTask.run_model(task, sample).values())
    if not torch.isfinite(loss):
        raise RuntimeError(f'non-finite loss at step {step}')
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    if step >= 10:
        times.append(duration)
        frames.append(int(sample['mel2ph'].gt(0).sum()))
if args.save_weights is not None:
    torch.save(model.state_dict(), args.save_weights)

result = {
    'model': args.model, 'gpu': torch.cuda.get_device_name(), 'precision': 'float32',
    'steps': args.steps, 'warmup': 10, 'workers': 0, 'accumulate_grad_batches': 1,
    'batch_size': 1, 'median_it_per_s': 1 / statistics.median(times),
    'median_frames_per_s': statistics.median(f / t for f, t in zip(frames, times)),
    'max_memory_allocated': torch.cuda.max_memory_allocated(),
    'frame_range': [min(frames), max(frames)],
}
print('BENCHMARK ' + json.dumps(result))
