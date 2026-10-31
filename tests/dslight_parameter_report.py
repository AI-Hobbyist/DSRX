import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd()))
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance, DiffSingerAllInOne
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary

def count(module):
    return sum(p.numel() for p in module.parameters())

def configure(light):
    set_hparams(config='configs/templates/all_in_one_lynxnet2_muon.yaml', print_hparams=False)
    if light:
        args = {'num_channels': 192, 'num_layers': 12, 'kernel_size': 7,
                'dilation_cycle': [1, 3, 9, 27, 81], 'cond_channels': 256,
                'time_embed_dim': 64, 'dropout_rate': 0.0}
        hparams['backbone_type'] = 'dslight'
        hparams['backbone_args'] = args.copy()
        hparams['pitch_prediction_args']['backbone_type'] = 'dslight'
        hparams['pitch_prediction_args']['backbone_args'] = args.copy()
        hparams['variances_prediction_args']['backbone_type'] = 'dslight'
        hparams['variances_prediction_args']['backbone_args'] = args.copy()
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'dslight'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = {
            'num_channels': 256, 'num_layers': 8, 'kernel_size': 7,
            'dilation_cycle': [1, 3, 9], 'dropout_rate': 0.0,
        }

def report(light):
    configure(light)
    vocab_size = len(load_phoneme_dictionary())
    model = DiffSingerAllInOne(vocab_size=vocab_size, out_dims=128)
    acoustic, variance = model.acoustic, model.variance
    result = {
        'variant': 'dslight_candidate' if light else 'legacy',
        'vocab_size': vocab_size,
        'acoustic_total': count(acoustic),
        'acoustic_encoder': count(acoustic.fs2),
        'acoustic_aux': count(acoustic.aux_decoder),
        'acoustic_backbone': count(acoustic.diffusion.velocity_fn),
        'variance_total': count(variance),
        'variance_encoder_including_duration': count(variance.fs2),
        'variance_pitch_backbone': count(variance.pitch_predictor.velocity_fn),
        'variance_parameter_backbone': count(variance.variance_predictor.velocity_fn),
        'all_in_one_total': count(model),
    }
    separate_acoustic = DiffSingerAcoustic(vocab_size=vocab_size, out_dims=128)
    separate_variance = DiffSingerVariance(vocab_size=vocab_size)
    result['separate_acoustic_total'] = count(separate_acoustic)
    result['separate_variance_total'] = count(separate_variance)
    assert count(model) == count(separate_acoustic) + count(separate_variance)
    if light:
        with torch.no_grad():
            for backbone, features, bins in ((variance.pitch_predictor.velocity_fn, 1, 64),
                                             (variance.variance_predictor.velocity_fn, 4, 12)):
                x = torch.randn(1, features, bins, 8)
                y = backbone(x, torch.tensor([0.5]), torch.randn(1, 256, 8))
                assert y.shape == x.shape and torch.isfinite(y).all()
    print('PARAMETERS', json.dumps(result))

report(False)
report(True)
