"""Run-scoped paired extraction records and complete VR harmonic cache."""

import hashlib
import json
import os
import pathlib
import pickle

import numpy as np
import yaml

from utils.hparams import hparams, mini_ddsp_config


def enabled():
    return mini_ddsp_config(hparams).get("enabled", False)


def cache_directory():
    directory = hparams.get("_mini_ddsp_cache_dir") or os.environ.get(
        "DSRX_MINI_DDSP_CACHE_DIR"
    )
    if not directory:
        raise RuntimeError("Mini-DDSP preprocessing must run through AllInOneBinarizer")
    return pathlib.Path(directory)


def pairing_id(name, key_shift=0.0, speed=1.0, spk_id=None):
    augmentation = json.dumps(
        [float(key_shift), float(speed), spk_id], separators=(",", ":")
    )
    digest = hashlib.sha256((name + augmentation).encode()).hexdigest()
    return digest, augmentation


def save_pair(item, waveform):
    directory = cache_directory() / "pairs" / item["_mini_ddsp_split"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (item["sample_id"].decode("ascii") + ".pkl")).open(
        "wb"
    ) as target:
        pickle.dump(
            {"item": item, "waveform": np.asarray(waveform, dtype=np.float32)}, target
        )


def harmonic_cache_path(waveform, sample_rate):
    model = pathlib.Path(hparams["hnsep_ckpt"])
    # The directory is unique to this run; file identity additionally prevents
    # reuse across differing separator settings inside a process.
    identity = [
        str(model.resolve()),
        model.stat().st_size,
        model.stat().st_mtime_ns,
        int(sample_rate),
        int(hparams.get("hnsep_sample_rate", 44100)),
        "mono-f32-v1",
    ]
    digest = hashlib.sha256(json.dumps(identity).encode())
    digest.update(model.with_name("config.yaml").read_bytes())
    digest.update(np.asarray(waveform, dtype=np.float32).tobytes())
    directory = cache_directory() / "harmonic"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (digest.hexdigest() + ".pkl")


def vr_work_sample_rate():
    model_config = pathlib.Path(hparams["hnsep_ckpt"]).with_name("config.yaml")
    metadata = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    declared = metadata.get("sr", metadata.get("sample_rate"))
    configured = hparams.get("hnsep_sample_rate", declared)
    if configured is None:
        raise ValueError("VR sample rate is missing: set shared hnsep_sample_rate")
    if declared is not None and int(configured) != int(declared):
        raise ValueError("hnsep_sample_rate disagrees with VR weight metadata")
    return int(configured)
