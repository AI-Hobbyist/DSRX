"""One waveform backend contract for all-in-one inference and validation."""

import json
import pathlib

import torch

from basics.base_vocoder import BaseVocoder
from utils.hparams import hparams, mini_ddsp_config


def validate_nsf_contract(config, metadata):
    fields = {
        "audio_sample_rate": "sampling_rate",
        "hop_size": "hop_size",
        "audio_num_mel_bins": "num_mels",
        "fft_size": "n_fft",
        "win_size": "win_size",
        "fmin": "fmin",
        "fmax": "fmax",
    }
    mismatches = [
        f"{key}={config[key]} (NSF {metadata.get(field)})"
        for key, field in fields.items()
        if config[key] != metadata.get(field)
    ]
    # The project's NSF implementation consumes natural-log mel. Its sidecar
    # historically omits mel_base, so absence means that established contract.
    if config.get("mel_base") != "e" or metadata.get("mel_base", "e") != "e":
        mismatches.append("mel_base must be natural log (e)")
    if mismatches:
        raise ValueError(
            "NSF-HiFiGAN audio contract mismatch: " + "; ".join(mismatches)
        )


class AllInOneVocoder(BaseVocoder):
    def __init__(self, backend, internal):
        self.backend = backend
        self.internal = internal

    def get_device(self):
        if self.internal:
            return next(self.backend.parameters()).device
        return self.backend.get_device()

    def to_device(self, device):
        if self.internal:
            self.backend.to(device)
        else:
            self.backend.to_device(device)

    @torch.inference_mode()
    def spec2wav_torch(self, mel, f0, **kwargs):
        if self.internal:
            return self.backend(mel, f0)
        # The historical external adapter flattens its batch. Calling each
        # item separately preserves the public [B,N] waveform contract.
        return torch.stack(
            [
                self.backend.spec2wav_torch(
                    mel[index : index + 1], f0=f0[index : index + 1], **kwargs
                )
                for index in range(mel.shape[0])
            ]
        )

    def spec2wav(self, mel, f0, **kwargs):
        mel = torch.as_tensor(mel, dtype=torch.float32, device=self.get_device())
        f0 = torch.as_tensor(f0, dtype=torch.float32, device=self.get_device())
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        if f0.ndim == 1:
            f0 = f0.unsqueeze(0)
        waveform = self.spec2wav_torch(mel, f0, **kwargs)
        return waveform.squeeze(0).cpu().numpy()


def build_embedded_nsf(metadata):
    from modules.nsf_hifigan.env import AttrDict
    from modules.nsf_hifigan.models import Generator

    generator = Generator(AttrDict(metadata))
    return generator.requires_grad_(False).eval()


class EmbeddedNsfVocoder(AllInOneVocoder):
    @torch.inference_mode()
    def spec2wav_torch(self, mel, f0, **kwargs):
        if f0.ndim == 3 and f0.shape[-1] == 1:
            f0 = f0.squeeze(-1)
        return self.backend(mel.transpose(1, 2), f0.float()).squeeze(1)


def build_all_in_one_vocoder(model, backend=None):
    if hparams["all_in_one"].get("embedded_nsf"):
        if backend not in (None, "nsf-hifigan"):
            raise ValueError("This AIO checkpoint contains an embedded NSF-HiFiGAN")
        return EmbeddedNsfVocoder(model.vocoder, internal=True)
    if not mini_ddsp_config(hparams).get("enabled", False):
        raise ValueError("All-in-one vocoder selection requires Mini-DDSP enabled")
    backend = backend or "mini-ddsp"
    if backend == "mini-ddsp":
        return AllInOneVocoder(model.vocoder, internal=True)
    if backend != "nsf-hifigan":
        raise ValueError(f"Unknown all-in-one waveform backend: {backend}")
    from modules.vocoders.nsf_hifigan import NsfHifiGAN

    metadata_path = pathlib.Path(hparams["vocoder_ckpt"]).with_name("config.json")
    validate_nsf_contract(
        hparams, json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    vocoder = AllInOneVocoder(NsfHifiGAN(), internal=False)
    vocoder.to_device(next(model.parameters()).device)
    return vocoder
