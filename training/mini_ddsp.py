"""ID-paired acoustic supervision and bounded joint waveform training."""

import pathlib

import h5py
import torch
from torch import nn
from torch.nn import functional as F

import utils
from training.acoustic_task import AcousticDataset
from utils.hparams import hparams, mini_ddsp_config
from utils.indexed_datasets import IndexedDataset


class PairedAcousticDataset(AcousticDataset):
    def __init__(self, prefix, preload=False, data_dir=None):
        super().__init__(prefix, preload=preload, data_dir=data_dir)
        vocoder_dir = pathlib.Path(data_dir).parent / "vocoder"
        message = (
            "Mini-DDSP requires paired vocoder data; rerun all-in-one preprocessing"
        )
        if not (vocoder_dir / f"{prefix}.data").exists():
            raise FileNotFoundError(message)
        self.vocoder_data = IndexedDataset(vocoder_dir, prefix)
        self.pair_indices = {}
        # Read only identifiers, rather than materializing all waveforms to
        # build a lookup table for a large training corpus.
        with h5py.File(vocoder_dir / f"{prefix}.data", "r") as source:
            for index in range(len(source)):
                if "sample_id" not in source[str(index)]:
                    raise ValueError(message)
                self.pair_indices[source[str(index)]["sample_id"][()].item()] = index
        with h5py.File(pathlib.Path(data_dir) / f"{prefix}.data", "r") as source:
            for index in range(len(source)):
                if "sample_id" not in source[str(index)]:
                    raise ValueError(message)
                identity = source[str(index)]["sample_id"][()].item()
                if identity not in self.pair_indices:
                    raise ValueError(f"{message}: missing {identity!r}")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        paired = self.vocoder_data[self.pair_indices[item["sample_id"]]]
        if paired["sample_rate"] != hparams["audio_sample_rate"]:
            raise ValueError(
                "Paired vocoder sample rate disagrees with training configuration"
            )
        if not torch.equal(item["mel"], paired["mel"]) or not torch.equal(
            item["f0"], paired["f0"]
        ):
            raise ValueError(
                f"Acoustic/vocoder features disagree for {item['sample_id']!r}"
            )
        item["vocoder"] = paired
        return item

    def collater(self, samples):
        batch = super().collater(samples)
        if not samples:
            return batch
        paired = [sample["vocoder"] for sample in samples]
        batch["vocoder"] = {
            "waveform": utils.collate_nd([item["waveform"] for item in paired], 0.0),
            "amplitude": utils.collate_nd([item["amplitude"] for item in paired], 0.0),
            "uv": utils.collate_nd([item["uv"] for item in paired], True),
            "sample_lengths": torch.tensor([item["sample_length"] for item in paired]),
            "frame_lengths": torch.tensor([item["length"] for item in paired]),
        }
        return batch


def joint_factor(step, settings):
    warmup = int(settings.get("warmup_steps", 2000))
    ramp = int(settings.get("joint_ramp_steps", 2000))
    if step < warmup:
        return 0.0
    return min(1.0, (step - warmup) / ramp) if ramp > 0 else 1.0


class MiniDDSPLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.settings = mini_ddsp_config(config)
        self.hop = int(config["hop_size"])
        self.segment = int(self.settings.get("segment_frames", 128))
        self.context = (
            int(self.settings.get("num_layers", 3))
            * (int(self.settings.get("kernel_size", 31)) // 2)
            + 1
        )
        window = int(config["win_size"])
        self.scales = sorted({max(16, window // 2), window, window * 2})
        for size in self.scales:
            self.register_buffer(
                f"window_{size}", torch.hann_window(size), persistent=False
            )

    def spectral(self, predicted, target):
        # Inputs are already cut to valid samples; padded frames/samples never
        # enter a statistic, including windows straddling the segment edges.
        losses = []
        for size in self.scales:
            window = getattr(self, f"window_{size}")
            kwargs = dict(
                n_fft=size,
                hop_length=max(1, size // 4),
                win_length=size,
                window=window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            prediction = torch.stft(predicted.float(), **kwargs).abs()
            reference = torch.stft(target.float(), **kwargs).abs()
            losses.append(
                (prediction - reference).abs().mean()
                + (
                    torch.log(prediction.clamp_min(1e-5))
                    - torch.log(reference.clamp_min(1e-5))
                )
                .abs()
                .mean()
            )
        return torch.stack(losses).mean()

    def forward(self, vocoder, sample, output, acoustic, step, training=True):
        paired = sample["vocoder"]
        weight = self.settings.get("loss", {})
        factor = joint_factor(step, self.settings)
        losses = {"ddsp_spectral_loss": [], "ddsp_amplitude_loss": []}
        if factor > 0:
            if output.clean_out is None:
                raise ValueError(
                    "Joint Mini-DDSP training requires the main acoustic predictor"
                )
            losses["ddsp_joint_loss"] = []
            if output.aux_out is not None:
                losses["ddsp_aux_joint_loss"] = []
        for index in range(sample["size"]):
            frames = int(paired["frame_lengths"][index])
            segment = min(self.segment, frames)
            start = (
                int(torch.randint(frames - segment + 1, ()).item())
                if training
                else (frames - segment) // 2
            )
            left, right = max(0, start - self.context), min(
                frames, start + segment + self.context
            )
            center = (start - left) * self.hop
            count = min(
                segment * self.hop,
                int(paired["sample_lengths"][index]) - start * self.hop,
            )
            if count <= 0:
                raise ValueError("Paired sample has no valid waveform samples")
            f0 = (
                sample["f0"][index : index + 1, left:right]
                * ~paired["uv"][index : index + 1, left:right]
            )
            target = paired["waveform"][
                index : index + 1, start * self.hop : start * self.hop + count
            ]
            actual = vocoder(
                sample["mel"][index : index + 1, left:right], f0, return_components=True
            )
            losses["ddsp_spectral_loss"].append(
                self.spectral(actual["waveform"][:, center : center + count], target)
            )
            first = start - left
            amplitude = actual["amplitude"][:, first : first + segment]
            amplitude_target = paired["amplitude"][
                index : index + 1, start : start + segment
            ]
            losses["ddsp_amplitude_loss"].append(
                (
                    torch.log(amplitude.clamp_min(1e-5))
                    - torch.log(amplitude_target.clamp_min(1e-5))
                )
                .abs()
                .mean()
            )
            if factor > 0:
                predicted = vocoder(output.clean_out[index : index + 1, left:right], f0)
                losses["ddsp_joint_loss"].append(
                    self.spectral(predicted[:, center : center + count], target)
                )
                if output.aux_out is not None:
                    aux_mel = acoustic.aux_decoder.denorm_spec(
                        output.aux_out[index : index + 1, left:right]
                    )
                    auxiliary = vocoder(aux_mel, f0)
                    losses["ddsp_aux_joint_loss"].append(
                        self.spectral(auxiliary[:, center : center + count], target)
                    )
        weights = {
            "ddsp_spectral_loss": weight.get("spectral_weight", 1.0),
            "ddsp_amplitude_loss": weight.get("amplitude_weight", 0.3),
            "ddsp_joint_loss": factor * weight.get("joint_weight", 1.0),
            "ddsp_aux_joint_loss": factor * weight.get("joint_weight", 1.0),
        }
        result = {
            name: torch.stack(values).mean() * weights[name]
            for name, values in losses.items()
        }
        if factor == 0:
            result["ddsp_joint_loss"] = sample["mel"].new_zeros(())
            if output.aux_out is not None:
                result["ddsp_aux_joint_loss"] = sample["mel"].new_zeros(())
        return result
