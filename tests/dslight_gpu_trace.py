import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd()))
from inference.optimization import optimize_model_for_inference
from modules.toplevel import DiffSingerAcoustic
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary

config = sys.argv[1] if len(sys.argv) > 1 else 'data/dslight_smoke/config.yaml'
weights = sys.argv[2] if len(sys.argv) > 2 else 'data/dslight_smoke/dslight_60_steps.pth'
set_hparams(config=config, print_hparams=False)
hparams['inference_warmup_frames'] = [16, 32]
torch.manual_seed(11)
model = DiffSingerAcoustic(len(load_phoneme_dictionary()), 128).cuda().eval()
model.load_state_dict(torch.load(weights, map_location='cuda', weights_only=True))
spec = torch.randn(1, 1, 128, 24, device='cuda')
step = torch.tensor([500.0], device='cuda')
condition = torch.randn(1, 256, 24, device='cuda')
with torch.inference_mode():
    reference = model.diffusion.velocity_fn(spec, step, condition)
report = optimize_model_for_inference(model, model_kind='acoustic', device='cuda')
with torch.inference_mode():
    actual = model.diffusion.velocity_fn(spec, step, condition)
assert report.torchscript and len(report.backbones) == 1 and report.backbones[0].error is None
assert torch.isfinite(actual).all()
torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
print('GPU velocity trace passed', report.to_dict())
