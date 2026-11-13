"""Independent indexed vocoder package from this run's paired source records."""

import json
import pickle
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from basics.base_binarizer import BaseBinarizer
from preprocessing.mini_ddsp_cache import cache_directory
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams, mini_ddsp_config

VOCODER_ATTRIBUTES = [
    "sample_id",
    "augmentation_id",
    "sample_rate",
    "sample_length",
    "waveform",
    "mel",
    "f0",
    "uv",
    "amplitude",
    "length",
    "spk_id",
]


def harmonic_amplitudes(waveform, f0, uv, device):
    """Windowed sinusoidal projection using a configured number of F0 cycles.

    Frame and harmonic groups bound memory; UV and above-Nyquist harmonics
    have zero amplitude. A unit cosine at a bin center has unit amplitude.
    """
    settings = mini_ddsp_config(hparams)
    sample_rate = hparams["audio_sample_rate"]
    hop = hparams["hop_size"]
    harmonics = settings.get("harmonics", 128)
    cycles = settings.get("harmonic_analysis_cycles", 4)
    pitch = torch.as_tensor(f0, device=device).float()
    half_width = torch.ceil(
        cycles * sample_rate / pitch.clamp_min(hparams["f0_min"]) / 2
    )
    radius = int(half_width.max().item())
    offsets = torch.arange(-radius, radius + 1, device=device).float()
    source = torch.as_tensor(waveform, device=device).float()
    source = F.pad(source, (radius, radius + hop))
    harmonic_indices = torch.arange(1, harmonics + 1, device=device).float()
    outputs = []
    for start in range(0, len(f0), 16):
        stop = min(start + 16, len(f0))
        centers = torch.arange(start, stop, device=device) * hop + radius
        indices = (centers[:, None] + offsets[None, :]).long()
        window = 0.5 + 0.5 * torch.cos(
            torch.pi * offsets[None, :] / half_width[start:stop, None]
        )
        window = window * (offsets.abs()[None, :] <= half_width[start:stop, None])
        windowed = source[indices] * window
        groups = []
        for group in range(0, harmonics, 16):
            frequencies = (
                pitch[start:stop, None] * harmonic_indices[None, group : group + 16]
            )
            angle = (2 * torch.pi / sample_rate) * frequencies[:, :, None] * offsets
            real = (torch.cos(angle) * windowed[:, None, :]).sum(-1)
            imag = (torch.sin(angle) * windowed[:, None, :]).sum(-1)
            amplitude = (
                2
                * torch.sqrt(real.square() + imag.square())
                / window.sum(-1, keepdim=True)
            )
            amplitude = amplitude * (frequencies < sample_rate / 2)
            groups.append(amplitude)
        outputs.append(torch.cat(groups, dim=1))
    result = torch.cat(outputs).cpu().numpy().astype(np.float32)
    result[np.asarray(uv, dtype=bool)] = 0
    return result


class VocoderBinarizer(BaseBinarizer):
    def __init__(self, datasets=None, binary_data_dir=None):
        super().__init__(
            datasets=datasets,
            data_attrs=VOCODER_ATTRIBUTES,
            binary_data_dir=binary_data_dir,
        )

    def process(self):
        self.binary_data_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (("spk_map", self.spk_map), ("lang_map", self.lang_map)):
            with (self.binary_data_dir / f"{name}.json").open(
                "w", encoding="utf-8"
            ) as target:
                json.dump(value, target, ensure_ascii=False)
        for lang, path in hparams["dictionaries"].items():
            shutil.copy(path, self.binary_data_dir / f"dictionary-{lang}.txt")
        with (self.binary_data_dir / "audio_contract.json").open(
            "w", encoding="utf-8"
        ) as target:
            json.dump(
                {
                    "version": 1,
                    "sample_rate": hparams["audio_sample_rate"],
                    "hop_size": hparams["hop_size"],
                    "mel_base": hparams["mel_base"],
                    "hnsep_sample_rate": hparams.get("hnsep_sample_rate", 44100),
                },
                target,
                indent=2,
            )
        for split in ("valid", "train"):
            files = sorted((cache_directory() / "pairs" / split).glob("*.pkl"))
            if not files:
                raise RuntimeError(
                    f"No paired {split} records; rerun all-in-one preprocessing"
                )
            self.items = {path.stem: {"record": path} for path in files}
            names = list(self.items)
            self._train_item_names = names if split == "train" else []
            self._valid_item_names = names if split == "valid" else []
            self.process_dataset(split)

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        with meta_data["record"].open("rb") as source:
            record = pickle.load(source)
        item, waveform = record["item"], record["waveform"]
        separator = DecomposedWaveform(
            waveform,
            samplerate=hparams["audio_sample_rate"],
            f0=item["f0"] * ~item["uv"],
            hop_size=hparams["hop_size"],
            fft_size=hparams["fft_size"],
            win_size=hparams["win_size"],
            algorithm="vr",
            device=self.device,
        )
        harmonic = separator.harmonic()
        item["amplitude"] = harmonic_amplitudes(
            harmonic, item["f0"], item["uv"], self.device
        )
        item["sample_rate"] = hparams["audio_sample_rate"]
        count = item["length"] * hparams["hop_size"]
        item["sample_length"] = min(len(waveform), count)
        item["waveform"] = np.pad(waveform[:count], (0, max(0, count - len(waveform))))
        return item
