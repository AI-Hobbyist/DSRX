"""ID-paired acoustic supervision and bounded joint waveform training."""

import json
import math
import pathlib
import random

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
        message = "Mini-DDSP requires original-audio vocoder data; run scripts/binarize_vocoder.py"
        if not (vocoder_dir / f"{prefix}.data").exists():
            raise FileNotFoundError(message)
        contract = json.loads(
            (vocoder_dir / "audio_contract.json").read_text(encoding="utf-8")
        )
        for key in (
            "audio_sample_rate",
            "audio_num_mel_bins",
            "hop_size",
            "fft_size",
            "win_size",
            "fmin",
            "fmax",
            "mel_base",
            "pe",
            "f0_min",
            "f0_max",
        ):
            if contract[key] != hparams[key]:
                raise ValueError(f"Vocoder audio contract disagrees on {key}")
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
        original = item["augmentation_id"] == b"[0.0,1.0,null]"
        if original and item["mel"].shape != paired["mel"].shape:
            raise ValueError(
                f"Original acoustic/vocoder frame lengths disagree for {item['sample_id']!r}"
            )
        item["vocoder"] = paired
        item["vocoder_paired"] = original
        return item

    def collater(self, samples):
        batch = super().collater(samples)
        if not samples:
            return batch
        paired = [sample["vocoder"] for sample in samples]
        batch["vocoder"] = {
            "mel": utils.collate_nd([item["mel"] for item in paired], 0.0),
            "f0": utils.collate_nd([item["f0"] for item in paired], 0.0),
            "waveform": utils.collate_nd([item["waveform"] for item in paired], 0.0),
            "amplitude": utils.collate_nd([item["amplitude"] for item in paired], 0.0),
            "uv": utils.collate_nd([item["uv"] for item in paired], True),
            "sample_lengths": torch.tensor([item["sample_length"] for item in paired]),
            "frame_lengths": torch.tensor([item["length"] for item in paired]),
            "paired": torch.tensor([sample["vocoder_paired"] for sample in samples]),
        }
        return batch


def joint_factor(step, settings):
    warmup = int(settings.get("warmup_steps", 2000))
    ramp = int(settings.get("joint_ramp_steps", 2000))
    if step < warmup:
        return 0.0
    return min(1.0, (step - warmup) / ramp) if ramp > 0 else 1.0


def augment_vocoder_volume(mel, waveform, amplitude, settings, training):
    """Reference Mini-DDSP log-gain augmentation with aligned amplitude labels."""
    if not training or not settings.get("volume_enabled", True):
        return mel, waveform, amplitude
    lower = float(settings.get("log_volume_min", -1.0))
    upper = float(settings.get("log_volume_max", 1.0))
    peak = float(waveform.abs().max()) + 1e-5
    log_gain = random.uniform(lower, min(upper, math.log(1.0 / peak)))
    gain = math.exp(log_gain)
    return mel + log_gain, waveform * gain, amplitude * gain


class MiniDDSPLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.settings = mini_ddsp_config(config)
        self.augmentation = self.settings.get("training_augmentation", {})
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

    @torch.no_grad()
    def validation_joint(self, vocoder, sample, output):
        """Score inference mel with full-utterance phase and a fixed center segment."""
        paired = sample["vocoder"]
        losses = {"ddsp_joint_loss": []}
        if output.aux_out is not None:
            losses["ddsp_aux_joint_loss"] = []
        for index in range(sample["size"]):
            if not paired["paired"][index]:
                raise ValueError(
                    "Joint validation requires original, unaugmented samples"
                )
            frames = int(paired["frame_lengths"][index])
            segment = min(self.segment, frames)
            start = ((frames - segment) // 2) * self.hop
            stop = min(start + segment * self.hop, int(paired["sample_lengths"][index]))
            target = paired["waveform"][index : index + 1, start:stop]
            f0 = sample["f0"][index : index + 1, :frames]
            predictions = {"ddsp_joint_loss": output.diff_out}
            if output.aux_out is not None:
                predictions["ddsp_aux_joint_loss"] = output.aux_out
            for name, mel in predictions.items():
                waveform = vocoder(mel[index : index + 1, :frames], f0)
                losses[name].append(self.spectral(waveform[:, start:stop], target))
        return {name: torch.stack(values).mean() for name, values in losses.items()}

    def forward(
        self,
        vocoder,
        sample,
        output,
        acoustic,
        step,
        training=True,
        return_metrics=False,
        include_joint=True,
    ):
        paired = sample["vocoder"]
        weight = self.settings.get("loss", {})
        factor = joint_factor(step, self.settings) if include_joint else 0.0
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
                if training and self.augmentation.get("random_crop", True)
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
            # Match inference's interpolated acoustic F0. Zero UV amplitude
            # targets must teach the controller to suppress harmonics rather
            # than bypassing that supervision by forcing its F0 to zero.
            f0 = paired["f0"][index : index + 1, left:right]
            target = paired["waveform"][
                index : index + 1, start * self.hop : start * self.hop + count
            ]
            first = start - left
            amplitude_target = paired["amplitude"][
                index : index + 1, start : start + segment
            ]
            mel, augmented_target, amplitude_target = augment_vocoder_volume(
                paired["mel"][index : index + 1, left:right],
                target,
                amplitude_target,
                self.augmentation,
                training,
            )
            actual = vocoder(mel, f0, return_components=True)
            losses["ddsp_spectral_loss"].append(
                self.spectral(
                    actual["waveform"][:, center : center + count], augmented_target
                )
            )
            amplitude = actual["amplitude"][:, first : first + segment]
            losses["ddsp_amplitude_loss"].append(
                (
                    torch.log(amplitude.clamp_min(1e-5))
                    - torch.log(amplitude_target.clamp_min(1e-5))
                )
                .abs()
                .mean()
            )
            if factor > 0 and paired["paired"][index]:
                # Independent F0 extraction can differ by floating-point roundoff.
                # Acoustic joint supervision uses its own conditioning curve.
                joint_f0 = sample["f0"][index : index + 1, left:right]
                predicted = vocoder(
                    output.clean_out[index : index + 1, left:right], joint_f0
                )
                losses["ddsp_joint_loss"].append(
                    self.spectral(predicted[:, center : center + count], target)
                )
                if output.aux_out is not None:
                    aux_mel = acoustic.aux_decoder.denorm_spec(
                        output.aux_out[index : index + 1, left:right]
                    )
                    auxiliary = vocoder(aux_mel, joint_f0)
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
            if values
        }
        if include_joint and "ddsp_joint_loss" not in result:
            result["ddsp_joint_loss"] = sample["mel"].new_zeros(())
            if output.aux_out is not None:
                result["ddsp_aux_joint_loss"] = sample["mel"].new_zeros(())
        if not return_metrics:
            return result
        joint_values = losses.get("ddsp_joint_loss", [])
        metrics = {
            "ddsp_joint_factor": factor * weight.get("joint_weight", 1.0),
            "ddsp_joint_sample_count": len(joint_values),
        }
        for name in ("ddsp_joint_loss", "ddsp_aux_joint_loss"):
            values = losses.get(name, [])
            if values:
                metrics[name.replace("_loss", "_raw_loss")] = (
                    torch.stack(values).mean().detach()
                )
        return result, metrics
