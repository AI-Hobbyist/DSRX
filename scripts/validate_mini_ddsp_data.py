"""Real opencpop three-package validation; never edits the provided assets."""

import argparse
import csv
import hashlib
import json
import pathlib
import random
import shutil
from unittest.mock import patch

import numpy as np
import torch
import yaml

from preprocessing.all_in_one_binarizer import AllInOneBinarizer
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--singing", choices=("on", "off"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    (raw / "wavs").mkdir(parents=True, exist_ok=True)
    source = pathlib.Path("data/opencpop")
    with (source / "transcriptions.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        all_rows = list(reader)
    holdout = "2033001224"
    rows = [row for row in all_rows if row["name"] != holdout][:2]
    rows += [next(row for row in all_rows if row["name"] == holdout)]
    source_files = [source / "transcriptions.csv"]
    with (raw / "transcriptions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        path = source / "wavs" / (row["name"] + ".wav")
        source_files.append(path)
        shutil.copyfile(path, raw / "wavs" / path.name)
    before = {str(path): digest(path) for path in source_files}
    config = yaml.safe_load(
        pathlib.Path("ckpt/dslight/config.yaml").read_text(encoding="utf-8")
    )
    config["datasets"] = [
        {
            "raw_data_dir": str(raw),
            "speaker": "opencpop",
            "spk_id": 0,
            "language": "zh",
            "artifact_level": 0,
            "test_prefixes": [holdout],
        }
    ]
    config["binary_data_dir"] = str(args.output / "binary")
    config["dictionaries"] = {"zh": "ckpt/dslight/dictionary-zh.txt"}
    config["binarization_args"].update(num_workers=0, allow_missing_phonemes=True)
    config["all_in_one"]["mini_ddsp"] = {
        "enabled": True,
        "harmonics": 128,
        "harmonic_analysis_cycles": 4,
    }
    config["hnsep_sample_rate"] = 44100
    config["hnsep"] = "vr"
    config["augmentation_args"]["random_pitch_shifting"].update(
        enabled=True, range=[-2, 2], scale=1
    )
    config["augmentation_args"]["random_time_stretching"].update(
        enabled=True, range=[0.8, 1.2], scale=3
    )
    for name in ("energy", "breathiness", "voicing", "tension"):
        config[f"use_{name}_embed"] = args.singing == "on"
        config[f"predict_{name}"] = args.singing == "on"
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    set_hparams(config=str(config_path), print_hparams=False)
    from modules.hnsep.vr.nets import CascadedNet

    original = CascadedNet.predict_from_audio
    calls = []

    def counted(model, audio, *positional, **keywords):
        calls.append(audio.shape[-1])
        return original(model, audio, *positional, **keywords)

    with patch.object(CascadedNet, "predict_from_audio", counted):
        AllInOneBinarizer().process()
    report = {
        "singing": args.singing,
        "seed": 42,
        "sources": before,
        "vr_calls": len(calls),
        "splits": {},
    }
    identities = {}
    for split in ("train", "valid"):
        acoustic = IndexedDataset(args.output / "binary/acoustic", split)
        variance = IndexedDataset(args.output / "binary/variance", split)
        vocoder = IndexedDataset(args.output / "binary/vocoder", split)
        paired = {
            vocoder[index]["sample_id"]: vocoder[index] for index in range(len(vocoder))
        }
        ids = set()
        transforms = []
        for index in range(len(acoustic)):
            item = acoustic[index]
            ids.add(item["sample_id"])
            target = paired[item["sample_id"]]
            torch.testing.assert_close(item["mel"], target["mel"], rtol=0, atol=0)
            torch.testing.assert_close(item["f0"], target["f0"], rtol=0, atol=0)
            assert (
                target["waveform"].numel()
                == target["mel"].shape[0] * config["hop_size"]
            )
            assert target["amplitude"].shape == (target["mel"].shape[0], 128)
            assert torch.isfinite(target["amplitude"]).all()
            transforms.append(json.loads(item["augmentation_id"]))
        assert ids == set(paired)
        identities[split] = ids
        report["splits"][split] = {
            "acoustic": len(acoustic),
            "variance": len(variance),
            "vocoder": len(vocoder),
            "unique_pairs": len(ids),
            "transforms": transforms,
        }
        # Close HDF5 handles before this process exits, also on Windows.
        for dataset in (acoustic, variance, vocoder):
            dataset.dset.close()
            dataset.dset = None
    assert not identities["train"] & identities["valid"]
    transforms = report["splits"]["train"]["transforms"]
    assert any(shift != 0 and speed == 1 for shift, speed, _ in transforms)
    assert any(shift == 0 and speed != 1 for shift, speed, _ in transforms)
    assert any(shift != 0 and speed != 1 for shift, speed, _ in transforms)
    assert report["vr_calls"] == sum(len(ids) for ids in identities.values())
    assert before == {str(path): digest(path) for path in source_files}
    report["source_unchanged"] = True
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
