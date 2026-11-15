"""Internal differentiable Mini-DDSP, with self-contained synthesis resources.

Harmonic frame synthesis follows Mini-DDSP's cosine/Hann overlap-add
algorithm (yxlllc, MIT); see docs/licenses/mini-ddsp.txt. Noise basis v1 uses
seeded, disjoint FFT bands at the configured sample rate, never external files.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.hparams import mini_ddsp_config


class ControlBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.project = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.expand(y), dim=1)
        y = self.project(F.silu(self.depthwise(y)))
        return x + y.transpose(1, 2)


class MiniDDSP(nn.Module):
    resource_version = 1

    def __init__(self, config):
        super().__init__()
        settings = mini_ddsp_config(config)
        self.sample_rate = int(config["audio_sample_rate"])
        self.hop_size = int(config["hop_size"])
        self.harmonics = int(settings.get("harmonics", 128))
        self.noise_bands = int(settings.get("noise_bands", 64))
        self.group_size = int(settings.get("harmonic_group_size", 16))
        channels = int(settings.get("hidden_size", 256))
        kernel = int(settings.get("kernel_size", 31))
        if (
            min(
                self.sample_rate,
                self.hop_size,
                self.harmonics,
                self.noise_bands,
                self.group_size,
                channels,
            )
            <= 0
            or kernel % 2 != 1
        ):
            raise ValueError(
                "Mini-DDSP dimensions must be positive; kernel must be odd"
            )
        self.mel_embed = nn.Linear(config["audio_num_mel_bins"], channels)
        self.phase_embed = nn.Linear(self.harmonics, channels)
        self.blocks = nn.ModuleList(
            [
                ControlBlock(channels, kernel)
                for _ in range(int(settings.get("num_layers", 3)))
            ]
        )
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, self.harmonics * 2 + self.noise_bands)
        self.register_buffer(
            "harmonic_indices", torch.arange(1, self.harmonics + 1).float()
        )
        self.register_buffer("window", torch.hann_window(2 * self.hop_size))
        self.register_buffer(
            "sample_offsets", torch.arange(-self.hop_size, self.hop_size).float()
        )
        self.register_buffer("noise_basis", self._make_noise_basis())
        self.register_buffer("synthesis_version", torch.tensor(self.resource_version))

    def _make_noise_basis(self):
        # At least one positive-frequency FFT bin per noise band.
        size = max(self.sample_rate, 2 * self.noise_bands)
        generator = torch.Generator().manual_seed(1729)
        spectrum = torch.fft.rfft(torch.randn(size, generator=generator))
        bins = torch.arange(spectrum.numel())
        bands = torch.div(
            bins * self.noise_bands, spectrum.numel(), rounding_mode="floor"
        )
        mask = bands[None, :] == torch.arange(self.noise_bands)[:, None]
        basis = torch.fft.irfft(spectrum[None, :] * mask, n=size)
        return basis / basis.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)

    def _overlap_add(self, frames):
        # Each frame is centered on its timestamp. Adjacent half-windows sum
        # without fold/scatter, including odd and non-traced dynamic lengths.
        right = frames[:, :, self.hop_size :]
        left = F.pad(frames[:, 1:, : self.hop_size], (0, 0, 0, 1))
        return (right + left).flatten(1)

    def synthesize_harmonics(self, amplitude, phase, frequencies, group_size=None):
        group_size = self.group_size if group_size is None else group_size
        frames = torch.zeros_like(amplitude[:, :, :1]) * self.window
        for start in range(0, self.harmonics, group_size):
            stop = min(start + group_size, self.harmonics)
            angle = (
                frequencies[:, :, start:stop, None]
                * (2 * math.pi / self.sample_rate)
                * self.sample_offsets
                + phase[:, :, start:stop, None]
            )
            frames = (
                frames
                + (torch.cos(angle) * amplitude[:, :, start:stop, None]).sum(2)
                * self.window
            )
        return self._overlap_add(frames)

    def forward(self, mel, f0, lengths=None, return_components=False):
        # Controls and synthesis use FP32 under mixed precision. Accumulate
        # phase in FP64: long FP32 cumsums differ across eager/ORT kernels and
        # can cross a modulo boundary, discontinuously changing phase inputs.
        f0 = f0.float()
        frequencies = f0[:, :, None] * self.harmonic_indices
        increments = (frequencies[:, 1:] + frequencies[:, :-1]) * (
            math.pi * self.hop_size / self.sample_rate
        )
        phase = F.pad(torch.cumsum(increments.double(), dim=1), (0, 0, 1, 0))
        phase = torch.remainder(phase, 2 * math.pi).float()
        if lengths is None:
            valid = torch.ones_like(f0, dtype=torch.bool)
        else:
            valid = (
                torch.arange(mel.shape[1], device=mel.device)[None, :]
                < lengths[:, None]
            )
        x = self.mel_embed(mel * valid[:, :, None]) + self.phase_embed(phase)
        for block in self.blocks:
            x = block(x * valid[:, :, None])
        controls = self.output(self.norm(x)).float()
        amplitude, offset, noise_amplitude = torch.split(
            controls, [self.harmonics, self.harmonics, self.noise_bands], dim=-1
        )
        amplitude = F.softplus(amplitude) / self.harmonics
        amplitude = (
            amplitude * (frequencies < self.sample_rate / 2) * (f0 > 0)[:, :, None]
        )
        amplitude = amplitude * valid[:, :, None]
        harmonic = self.synthesize_harmonics(amplitude, phase + offset, frequencies)
        noise_amplitude = F.softplus(noise_amplitude) / self.noise_bands
        noise_amplitude = noise_amplitude * valid[:, :, None]
        count = mel.shape[1] * self.hop_size
        indices = torch.arange(count, device=mel.device) % self.noise_basis.shape[1]
        basis = self.noise_basis[:, indices]
        envelope = F.interpolate(
            noise_amplitude.transpose(1, 2),
            size=count,
            mode="linear",
            align_corners=False,
        )
        # Pitch-synchronous triangular noise envelope, fading to unmodulated
        # noise for unvoiced frames (the Mini-DDSP excitation convention).
        voiced_ratio = amplitude[:, :, 0] / (
            amplitude[:, :, 0] + noise_amplitude.sum(-1) + 1e-7
        )
        voiced_ratio = F.interpolate(
            voiced_ratio[:, None, :], size=count, mode="linear", align_corners=False
        ).squeeze(1)
        sample_f0 = F.interpolate(
            f0[:, None, :], size=count, mode="linear", align_corners=False
        ).squeeze(1)
        cycles = torch.remainder(
            torch.cumsum(sample_f0.double() / self.sample_rate, dim=1), 1
        ).float()
        triangle = torch.where(cycles < 0.2, cycles / 0.2, (1 - cycles) / 0.8)
        noise = (envelope * basis[None, :, :]).sum(1)
        noise = noise * (1 - voiced_ratio + voiced_ratio * triangle)
        waveform = harmonic + noise
        if lengths is not None:
            mask = (
                torch.arange(count, device=mel.device)[None, :]
                < lengths[:, None] * self.hop_size
            )
            waveform = waveform * mask
        if return_components:
            return {
                "waveform": waveform,
                "harmonic": harmonic,
                "noise": noise,
                "amplitude": amplitude,
            }
        return waveform
