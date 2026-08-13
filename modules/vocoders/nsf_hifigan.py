import pathlib
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ModuleNotFoundError:
    rank_zero_info = print

from modules.nsf_hifigan.models import load_model
from basics.base_vocoder import BaseVocoder
from modules.vocoders.registry import register_vocoder
from utils.hparams import hparams


DEFAULT_INFERENCE_BUCKETS = (
    128, 256, 384, 512, 768, 1024, 1536,
    2048, 3072, 4096, 6144, 8192, 10240,
)
DEFAULT_WARMUP_BUCKETS = (128, 256, 384, 512, 768)


def _parse_frames(value, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        value = default
    elif isinstance(value, str):
        value = [part.strip() for part in value.split('|') if part.strip()]
    elif isinstance(value, int):
        value = [value]
    result = sorted({int(item) for item in value if int(item) > 0})
    return tuple(result or default)


@register_vocoder
class NsfHifiGAN(BaseVocoder):
    def __init__(self):
        model_path = pathlib.Path(hparams['vocoder_ckpt'])
        if not model_path.exists():
            raise FileNotFoundError(
                f'NSF-HiFiGAN vocoder model is not found at \'{model_path}\'. '
                'Please follow instructions in docs/BestPractices.md#vocoders to get one.'
            )
        rank_zero_info(f'| Load HifiGAN: {model_path}')
        self.model, self.h = load_model(model_path)
        self._optimization_enabled = bool(hparams.get('inference_optimization', True))
        self._bucketed_backend_requested = bool(
            hparams.get('inference_vocoder_nsf_bucketed_backend', True)
        )
        self._selective_fp16 = False
        self._buckets = _parse_frames(
            hparams.get('inference_vocoder_nsf_buckets'), DEFAULT_INFERENCE_BUCKETS
        )
        self._warmup_buckets = _parse_frames(
            hparams.get('inference_vocoder_nsf_warmup_frames'), DEFAULT_WARMUP_BUCKETS
        )
        self._warmed_buckets: tuple[int, ...] = ()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to_device(self, device):
        resolved = torch.device(device)
        self.model.to(resolved)
        self._selective_fp16 = (
            self._optimization_enabled
            and resolved.type == 'cuda'
            and bool(hparams.get('inference_vocoder_nsf_selective_fp16', True))
        )
        if self._selective_fp16:
            self.model.half()
        else:
            self.model.float()
        rank_zero_info(
            '| vocoder optimization: NSF-HiFiGAN '
            f'selective_fp16={self._selective_fp16}, phase_dtype=fp32'
        )

    def get_device(self):
        return self.device

    def _warn_mismatch(self):
        if self.h.sampling_rate != hparams['audio_sample_rate']:
            print('Mismatch parameters: hparams[\'audio_sample_rate\']=', hparams['audio_sample_rate'], '!=',
                  self.h.sampling_rate, '(vocoder)')
        if self.h.num_mels != hparams['audio_num_mel_bins']:
            print('Mismatch parameters: hparams[\'audio_num_mel_bins\']=', hparams['audio_num_mel_bins'], '!=',
                  self.h.num_mels, '(vocoder)')
        if self.h.n_fft != hparams['fft_size']:
            print('Mismatch parameters: hparams[\'fft_size\']=', hparams['fft_size'], '!=', self.h.n_fft, '(vocoder)')
        if self.h.win_size != hparams['win_size']:
            print('Mismatch parameters: hparams[\'win_size\']=', hparams['win_size'], '!=', self.h.win_size,
                  '(vocoder)')
        if self.h.hop_size != hparams['hop_size']:
            print('Mismatch parameters: hparams[\'hop_size\']=', hparams['hop_size'], '!=', self.h.hop_size,
                  '(vocoder)')
        if self.h.fmin != hparams['fmin']:
            print('Mismatch parameters: hparams[\'fmin\']=', hparams['fmin'], '!=', self.h.fmin, '(vocoder)')
        if self.h.fmax != hparams['fmax']:
            print('Mismatch parameters: hparams[\'fmax\']=', hparams['fmax'], '!=', self.h.fmax, '(vocoder)')

    def _select_bucket(self, frames: int) -> int:
        return next((bucket for bucket in self._buckets if bucket >= frames), frames)

    @property
    def _bucketed_backend(self) -> bool:
        return (
            self._optimization_enabled
            and self._bucketed_backend_requested
            and self.device.type == 'cuda'
        )

    def get_optimization_info(self):
        return {
            'name': 'NsfHifiGAN',
            'enabled': self._optimization_enabled and self.device.type == 'cuda',
            'selective_fp16': self._selective_fp16,
            'phase_dtype': 'float32',
            'bucketed_backend': self._bucketed_backend,
            'backend_buckets': list(self._buckets),
            'warmed_buckets': list(self._warmed_buckets),
        }

    @torch.inference_mode()
    def warmup(self, frames: Optional[Iterable[int]] = None):
        if not self._bucketed_backend:
            return self.get_optimization_info()
        requested = _parse_frames(frames, self._warmup_buckets)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        try:
            for frame_count in requested:
                mel = torch.zeros(
                    1, frame_count, self.h.num_mels,
                    device=self.device, dtype=torch.float32,
                )
                f0 = torch.zeros(1, frame_count, device=self.device, dtype=torch.float32)
                self.spec2wav_torch(mel, f0=f0, use_vocoder_buckets=False)
                torch.cuda.synchronize(self.device)
            self._warmed_buckets = requested
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
        return self.get_optimization_info()

    @torch.inference_mode()
    def spec2wav_torch(self, mel, **kwargs):  # mel: [B, T, bins]
        self._warn_mismatch()
        if mel.dim() != 3:
            raise ValueError(f'NSF-HiFiGAN expects 3-D mel, got shape {mel.shape}.')
        original_frames = int(mel.shape[1])
        use_buckets = (
            self._bucketed_backend
            and bool(kwargs.get('use_vocoder_buckets', False))
            and mel.shape[0] == 1
        )
        if use_buckets:
            bucket_frames = self._select_bucket(original_frames)
            if bucket_frames > original_frames:
                mel = F.pad(mel, (0, 0, 0, bucket_frames - original_frames))
        c = mel.transpose(2, 1)
        mel_base = hparams.get('mel_base', 10)
        if mel_base != 'e':
            assert mel_base in [10, '10'], "mel_base must be 'e', '10' or 10."
            # log10 to log mel
            c = 2.30259 * c
        f0 = kwargs.get('f0')
        if f0 is None:
            raise ValueError('NSF-HiFiGAN requires f0 input.')
        if f0.dim() == 3 and f0.shape[-1] == 1:
            f0 = f0.squeeze(-1)
        f0 = f0.to(self.device, dtype=torch.float32)
        if use_buckets and f0.shape[1] < mel.shape[1]:
            f0 = F.pad(f0, (0, mel.shape[1] - f0.shape[1]))
        y = self.model(c.to(self.device), f0).view(-1)
        if use_buckets:
            y = y[:original_frames * int(self.h.hop_size)]
        return y

    def spec2wav(self, mel, **kwargs):
        mel_t = torch.as_tensor(mel, dtype=torch.float32, device=self.device)
        if mel_t.dim() == 2:
            mel_t = mel_t.unsqueeze(0)
        f0 = kwargs.get('f0')
        if f0 is None:
            raise ValueError('NSF-HiFiGAN requires f0 input.')
        f0_t = torch.as_tensor(f0, dtype=torch.float32, device=self.device)
        if f0_t.dim() == 1:
            f0_t = f0_t.unsqueeze(0)
        y = self.spec2wav_torch(mel_t, f0=f0_t)
        return y.cpu().numpy()
