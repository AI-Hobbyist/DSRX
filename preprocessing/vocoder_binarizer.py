"""Standalone audio-only Mini-DDSP binary preprocessing."""

import json
import os
from pathlib import Path
import tempfile

import librosa
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from modules.pe import initialize_pe
from preprocessing.mini_ddsp_cache import pairing_id
from utils.binarizer_utils import get_mel_torch
from utils.binarization_checkpoint import BinarizationCheckpoint, atomic_pickle
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams, mini_ddsp_config
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run

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
pitch_extractor = None


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


class VocoderBinarizer:
    """Read audio directly; no phoneme dictionaries, CSV or acoustic package."""

    def __init__(self):
        self.datasets = hparams["datasets"]
        self.binary_data_dir = Path(hparams["binary_data_dir"]) / "vocoder"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collect_sources(self):
        sources = {"train": [], "valid": []}
        identities = set()
        for dataset in self.datasets:
            directory = (
                Path(dataset["audio_dir"])
                if "audio_dir" in dataset
                else Path(dataset["raw_data_dir"]) / "wavs"
            )
            prefixes = tuple(str(value) for value in dataset.get("test_prefixes", []))
            paths = sorted(directory.rglob("*.wav"))
            names = {
                path.relative_to(directory).with_suffix("").as_posix() for path in paths
            }
            # Match DiffSinger: an exact name takes precedence over a prefix.
            valid_names = set()
            for prefix in prefixes:
                valid_names.update(
                    {prefix}
                    if prefix in names
                    else {name for name in names if name.startswith(prefix)}
                )
            for path in paths:
                name = path.relative_to(directory).with_suffix("").as_posix()
                identity, augmentation = pairing_id(f"{dataset['speaker']}:{name}")
                if identity in identities:
                    raise ValueError(
                        f"Duplicate vocoder recording: {dataset['speaker']}:{name}"
                    )
                identities.add(identity)
                split = "valid" if name in valid_names else "train"
                sources[split].append(
                    {
                        "path": str(path),
                        "sample_id": identity,
                        "augmentation_id": augmentation,
                        "spk_id": dataset.get("spk_id", 0),
                    }
                )
        for split, items in sources.items():
            if not items:
                raise ValueError(f"No {split} audio found for vocoder preprocessing")
        return sources

    def process(self):
        sources = self.collect_sources()
        self.binary_data_dir.mkdir(parents=True, exist_ok=True)
        contract_keys = (
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
        )
        contract = {key: hparams[key] for key in contract_keys}
        contract["version"] = 2
        # Cache only intermediate harmonic signals, then remove them. The final
        # waveform and labels live together in the indexed binary package.
        with tempfile.TemporaryDirectory(
            prefix=".vocoder-", dir=self.binary_data_dir
        ) as cache:
            previous = os.environ.get("DSRX_MINI_DDSP_CACHE_DIR")
            os.environ["DSRX_MINI_DDSP_CACHE_DIR"] = cache
            try:
                for split, items in sources.items():
                    self.process_dataset(split, items)
            finally:
                if previous is None:
                    os.environ.pop("DSRX_MINI_DDSP_CACHE_DIR", None)
                else:
                    os.environ["DSRX_MINI_DDSP_CACHE_DIR"] = previous
        temporary = self.binary_data_dir / "audio_contract.json.tmp"
        temporary.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        os.replace(temporary, self.binary_data_dir / "audio_contract.json")

    def process_dataset(self, split, sources):
        checkpoint = BinarizationCheckpoint(
            self.binary_data_dir,
            split,
            sources,
            VOCODER_ATTRIBUTES,
            {"waveform", "mel", "f0", "uv", "amplitude", "sample_id"},
            {source["sample_id"].encode("ascii") for source in sources},
        )
        if checkpoint.prepare():
            return
        start_index = checkpoint.state["next_index"]
        workers = int(hparams.get("binarization_args", {}).get("num_workers", 0))
        args = [[source] for source in sources[start_index:]]
        results = (
            chunked_multiprocess_run(self.process_item, args, num_workers=workers)
            if workers > 0 and args
            else (self.process_item(*arguments) for arguments in args)
        )
        builder = IndexedDatasetBuilder(
            self.binary_data_dir,
            split,
            VOCODER_ATTRIBUTES,
            resume=checkpoint.restoring,
        )
        if checkpoint.restoring:
            builder.truncate(checkpoint.state["count"])
            metadata = checkpoint.state["payload"]
        else:
            metadata = {"lengths": [], "names": [], "spk_ids": []}
            checkpoint.commit(builder, 0, metadata)
        processed = start_index
        try:
            for index, item in enumerate(
                tqdm(
                    results,
                    total=len(sources),
                    initial=start_index,
                    desc=f"vocoder/{split}",
                ),
                start=start_index,
            ):
                source = sources[index]
                builder.add_item(item)
                metadata["lengths"].append(item["length"])
                metadata["names"].append(source["path"])
                metadata["spk_ids"].append(item["spk_id"])
                processed = index + 1
                if processed % checkpoint.interval == 0:
                    checkpoint.commit(builder, processed, metadata)
            checkpoint.commit(builder, processed, metadata)
        finally:
            builder.finalize()
        atomic_pickle(self.binary_data_dir / f"{split}.meta", metadata)
        checkpoint.finish()

    @torch.no_grad()
    def process_item(self, source):
        waveform, _ = librosa.load(
            source["path"], sr=hparams["audio_sample_rate"], mono=True
        )
        mel = get_mel_torch(
            waveform,
            hparams["audio_sample_rate"],
            num_mel_bins=hparams["audio_num_mel_bins"],
            hop_size=hparams["hop_size"],
            win_size=hparams["win_size"],
            fft_size=hparams["fft_size"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            device=self.device,
        )
        global pitch_extractor
        if pitch_extractor is None:
            pitch_extractor = initialize_pe(self.device)
        f0, uv = pitch_extractor.get_pitch(
            waveform,
            samplerate=hparams["audio_sample_rate"],
            length=len(mel),
            hop_size=hparams["hop_size"],
            f0_min=hparams["f0_min"],
            f0_max=hparams["f0_max"],
            interp_uv=True,
        )
        if uv.all():
            raise ValueError(
                f"Vocoder audio contains no voiced frames: {source['path']}"
            )
        separator = DecomposedWaveform(
            waveform,
            samplerate=hparams["audio_sample_rate"],
            f0=f0 * ~uv,
            hop_size=hparams["hop_size"],
            fft_size=hparams["fft_size"],
            win_size=hparams["win_size"],
            algorithm=hparams["hnsep"],
            device=self.device,
        )
        amplitude = harmonic_amplitudes(separator.harmonic(), f0, uv, self.device)
        count = len(mel) * hparams["hop_size"]
        return {
            "sample_id": np.bytes_(source["sample_id"]),
            "augmentation_id": np.bytes_(source["augmentation_id"]),
            "spk_id": source["spk_id"],
            "sample_rate": hparams["audio_sample_rate"],
            "sample_length": min(len(waveform), count),
            "length": len(mel),
            "waveform": np.pad(waveform[:count], (0, max(0, count - len(waveform)))),
            "mel": mel,
            "f0": f0.astype(np.float32),
            "uv": uv.astype(bool),
            "amplitude": amplitude,
        }
