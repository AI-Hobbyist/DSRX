from collections import OrderedDict

from torch import nn

from basics.base_module import CategorizedModule
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from utils.hparams import hparams


class DiffSingerAllInOne(CategorizedModule):
    @property
    def category(self):
        return 'all_in_one'

    def __init__(self, vocab_size, out_dims):
        super().__init__()
        self.train_variance_module = any([
            hparams.get('predict_dur', False),
            hparams.get('predict_pitch', False),
            hparams.get('predict_energy', False),
            hparams.get('predict_breathiness', False),
            hparams.get('predict_voicing', False),
            hparams.get('predict_tension', False),
        ])
        self.train_acoustic_module = hparams.get('all_in_one', {}).get('train_acoustic', True)

        if self.train_variance_module:
            self.variance = DiffSingerVariance(vocab_size=vocab_size)
        else:
            self.variance = None
        if self.train_acoustic_module:
            self.acoustic = DiffSingerAcoustic(vocab_size=vocab_size, out_dims=out_dims)
        else:
            self.acoustic = None
        self.shared_encoder = hparams.get('all_in_one', {}).get('shared_encoder', False)
        if self.shared_encoder:
            self.share_encoder()
        self.apply_training_scopes()

    @staticmethod
    def _freeze_module(module):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def share_encoder(self):
        if self.variance is None or self.acoustic is None:
            raise ValueError('all_in_one.shared_encoder requires both variance and acoustic modules to be enabled.')
        if not hasattr(self.variance, 'fs2') or not hasattr(self.acoustic, 'fs2'):
            raise ValueError('all_in_one.shared_encoder requires both modules to expose fs2.')
        if not hasattr(self.variance.fs2, 'encoder') or not hasattr(self.acoustic.fs2, 'encoder'):
            raise ValueError('all_in_one.shared_encoder requires both fs2 modules to expose encoder.')
        self.acoustic.fs2.encoder = self.variance.fs2.encoder

    def apply_training_scopes(self):
        cfg = hparams.get('all_in_one', {})
        if self.acoustic is not None and not cfg.get('train_acoustic', True):
            self._freeze_module(self.acoustic)
        if self.variance is None:
            return
        train_variance = cfg.get('train_variance', True)
        if not train_variance and hparams.get('predict_dur', False):
            for module_name in ('dur_predictor', 'midi_embed', 'onset_embed', 'word_dur_embed'):
                self._freeze_module(getattr(self.variance.fs2, module_name, None))
        if not train_variance and hparams.get('predict_pitch', False):
            for module_name in (
                    'pitch_predictor', 'melody_encoder', 'delta_pitch_embed',
                    'base_pitch_embed', 'pitch_retake_embed'
            ):
                self._freeze_module(getattr(self.variance, module_name, None))
        if not train_variance:
            for module_name in ('variance_predictor', 'variance_embeds', 'pitch_embed'):
                self._freeze_module(getattr(self.variance, module_name, None))

    def named_submodules_for_summary(self):
        modules = OrderedDict()
        if self.variance is not None:
            if hasattr(self.variance, 'fs2'):
                modules['variance_encoder'] = self.variance.fs2.encoder
                if hasattr(self.variance.fs2, 'dur_predictor'):
                    modules['variance_dur'] = self.variance.fs2.dur_predictor
            if hasattr(self.variance, 'pitch_predictor'):
                modules['variance_pitch'] = self.variance.pitch_predictor
            if hasattr(self.variance, 'variance_predictor'):
                modules['variance_params'] = self.variance.variance_predictor
        if self.acoustic is not None:
            modules['acoustic'] = self.acoustic
        return modules

    def forward_variance(self, *args, **kwargs):
        if self.variance is None:
            return None, None, None
        return self.variance(*args, **kwargs)

    def forward_acoustic(self, *args, **kwargs):
        if self.acoustic is None:
            return None
        return self.acoustic(*args, **kwargs)
