"""Resume indexed preprocessing without storing separate feature files."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import random

import h5py

from utils.hparams import hparams


def atomic_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as target:
        pickle.dump(value, target, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def preprocessing_signature(sources, attributes):
    keys = {
        "datasets",
        "dictionaries",
        "augmentation_args",
        "pe",
        "pe_ckpt",
        "hnsep",
        "hnsep_ckpt",
        "hnsep_sample_rate",
        "hop_size",
        "fft_size",
        "win_size",
        "fmin",
        "fmax",
        "mel_base",
        "glide_types",
    }
    settings = {
        key: value
        for key, value in hparams.items()
        if key in keys
        or key.startswith(("audio_", "f0_", "use_", "predict_"))
        or key.endswith("_smooth_width")
    }
    settings["binarization_args"] = {
        key: value
        for key, value in hparams.get("binarization_args", {}).items()
        if key
        not in {
            "resume",
            "checkpoint_interval",
            "num_workers",
            "num_workers_per_gpu",
            "workers_per_gpu",
        }
    }
    mini = hparams.get("all_in_one", {}).get("mini_ddsp", {})
    settings["mini_ddsp"] = {
        key: mini.get(key)
        for key in ("enabled", "harmonics", "harmonic_analysis_cycles")
    }
    if "waveform" in attributes:
        # An audio-only package must remain independent of symbolic/model flags.
        settings = {
            key: value
            for key, value in settings.items()
            if key.startswith(("audio_", "f0_"))
            or key
            in {
                "pe",
                "pe_ckpt",
                "hnsep",
                "hnsep_ckpt",
                "hnsep_sample_rate",
                "hop_size",
                "fft_size",
                "win_size",
                "fmin",
                "fmax",
                "mel_base",
            }
        }
        settings["mini_ddsp"] = {
            key: mini.get(key) for key in ("harmonics", "harmonic_analysis_cycles")
        }
    paths = {source.get("wav_fn", source.get("path")) for source in sources}
    if "waveform" not in attributes:
        paths.update(hparams.get("dictionaries", {}).values())
        for dataset in hparams.get("datasets", []):
            raw_directory = Path(dataset["raw_data_dir"])
            paths.add(str(raw_directory / "transcriptions.csv"))
            if hparams.get("binarization_args", {}).get("prefer_ds", False):
                paths.update(str(path) for path in (raw_directory / "ds").glob("*.ds"))
    paths.update(hparams.get(key) for key in ("pe_ckpt", "hnsep_ckpt"))
    paths.update(
        str(Path(hparams[key]).with_name("config.yaml"))
        for key in ("pe_ckpt", "hnsep_ckpt")
        if hparams.get(key)
    )
    files = {}
    for name in sorted(path for path in paths if path):
        path = Path(name)
        if path.exists():
            status = path.stat()
            files[str(path.resolve())] = [status.st_size, status.st_mtime_ns]
    payload = {
        "version": 1,
        "settings": settings,
        "sources": sources,
        "attributes": attributes,
        "files": files,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


class BinarizationCheckpoint:
    def __init__(
        self, directory, split, sources, attributes, required, expected_ids=None
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.data_path = self.directory / f"{split}.data"
        self.meta_path = self.directory / f"{split}.meta"
        self.progress_path = self.directory / f"{split}.progress.pkl"
        self.complete_path = self.directory / f"{split}.complete.json"
        self.signature = preprocessing_signature(sources, attributes)
        self.required = set(required)
        self.expected_ids = expected_ids
        self.source_count = len(sources)
        self.interval = max(
            1, int(hparams.get("binarization_args", {}).get("checkpoint_interval", 16))
        )
        self.restoring = False
        self.state = {
            "signature": self.signature,
            "next_index": 0,
            "count": 0,
            "payload": None,
            "rng_state": random.getstate(),
        }

    def _validate_binary(self, check_sources=True):
        if not self.data_path.exists() or not self.meta_path.exists():
            return False
        with self.meta_path.open("rb") as source:
            metadata = pickle.load(source)
        with h5py.File(self.data_path, "r") as source:
            count = len(source)
            if not count or count != len(metadata["lengths"]):
                return False
            if set(source) != {str(index) for index in range(count)}:
                return False
            original_ids = set()
            for group in source.values():
                if not self.required.issubset(group):
                    return False
                if self.expected_ids is not None:
                    if "sample_id" not in group or "augmentation_id" not in group:
                        return False
                    if group["augmentation_id"][()].item() == b"[0.0,1.0,null]":
                        original_ids.add(group["sample_id"][()].item())
            if not check_sources:
                return True
            if self.expected_ids is not None:
                return original_ids == self.expected_ids
            return count >= self.source_count

    def prepare(self):
        resume = hparams.get("binarization_args", {}).get("resume", False)
        if resume and self.progress_path.exists():
            with self.progress_path.open("rb") as source:
                state = pickle.load(source)
            if state["signature"] != self.signature:
                raise ValueError(
                    f"Preprocessing inputs/config changed: {self.progress_path}; rerun without --resume"
                )
            if not self.data_path.exists():
                raise FileNotFoundError(f"Checkpoint binary missing: {self.data_path}")
            self.state = state
            self.restoring = True
            print(
                f"| Resume {self.directory.name}/{self.split}: {state['next_index']}/{self.source_count} sources",
                flush=True,
            )
        elif resume and self.complete_path.exists():
            completed = json.loads(self.complete_path.read_text(encoding="utf-8"))
            if completed["signature"] != self.signature:
                raise ValueError(
                    f"Preprocessing inputs/config changed: {self.complete_path}; rerun without --resume"
                )
            if not self._validate_binary(check_sources=False):
                raise ValueError(f"Completed binary is incomplete: {self.data_path}")
            print(f"| Reuse completed {self.directory.name}/{self.split}", flush=True)
            return True
        elif resume and self.data_path.exists() and self.meta_path.exists():
            if not self._validate_binary():
                raise ValueError(
                    f"Cannot reuse existing binary: {self.data_path}; rerun this stage without --resume"
                )
            # Older releases wrote .meta only after finishing a split. --resume
            # explicitly reuses these complete packages under the current config.
            print(
                f"| Reuse existing completed {self.directory.name}/{self.split} (legacy package; no original config fingerprint)",
                flush=True,
            )
            self.finish()
            return True
        else:
            self.complete_path.unlink(missing_ok=True)
        return False

    def commit(self, builder, next_index, payload):
        builder.dset.flush()
        self.state.update(next_index=next_index, count=builder.counter, payload=payload)
        atomic_pickle(self.progress_path, self.state)

    def finish(self):
        temporary = self.complete_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"signature": self.signature}), encoding="utf-8"
        )
        os.replace(temporary, self.complete_path)
        self.progress_path.unlink(missing_ok=True)
