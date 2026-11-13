"""Exercise the formal all-in-one CLI on a real held-out opencpop score."""

import argparse
import csv
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import yaml


def heldout_score():
    with pathlib.Path("data/opencpop/transcriptions.csv").open(
        encoding="utf-8"
    ) as source:
        row = next(
            item for item in csv.DictReader(source) if item["name"] == "2033001224"
        )
    phones = np.asarray(row["ph_dur"].split(), dtype=float)
    counts = np.asarray(row["ph_num"].split(), dtype=int)
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    word_durations = np.asarray(
        [
            phones[left:right].sum()
            for left, right in zip(boundaries[:-1], boundaries[1:])
        ]
    )
    word_starts = np.concatenate(([0.0], np.cumsum(word_durations)[:-1]))
    note_durations = np.asarray(row["note_dur"].split(), dtype=float)
    note_starts = np.concatenate(([0.0], np.cumsum(note_durations)[:-1]))
    slurs = [int(np.abs(word_starts - time).min() > 0.005) for time in note_starts]
    assert slurs.count(0) == len(counts), "Score word/note boundaries do not align"
    return {
        "ph_seq": row["ph_seq"],
        "ph_num": row["ph_num"],
        "note_seq": row["note_seq"],
        "note_dur": row["note_dur"],
        "note_slur": " ".join(map(str, slurs)),
        "lang": "zh",
    }, note_durations.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/mini_ddsp_m3/model_ckpt_steps_12.ckpt"),
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("artifacts/mini_ddsp_m4")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = (args.output / "runtime").resolve()
    subprocess.run(
        [
            sys.executable,
            "scripts/export_inference_ckpt.py",
            "--input",
            str(args.checkpoint),
            "--output",
            str(runtime),
            "--manifest",
            str(args.output / "checkpoint.json"),
            "--force",
        ],
        check=True,
    )
    config_path = runtime / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    external_path = config["vocoder_ckpt"]
    absent = runtime / "external-vocoder-not-provided.ckpt"
    assert not absent.exists()
    config["vocoder_ckpt"] = str(absent)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    assets = pathlib.Path(config["binary_data_dir"]) / "acoustic"
    for name in ("spk_map.json", "lang_map.json", "dictionary-zh.txt"):
        shutil.copyfile(assets / name, runtime / name)
    score, duration = heldout_score()
    project = args.output / "heldout.ds"
    segments = [
        {**score, "offset": offset}
        for offset in (0.10, duration + 0.35, 2 * duration + 0.20)
    ]
    project.write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base = [
        sys.executable,
        "scripts/infer.py",
        "all-in-one",
        str(project),
        "--exp",
        str(runtime),
        "--out",
        str(args.output),
        "--seed",
        "42",
        "--steps",
        "20",
    ]
    subprocess.run([*base, "--title", "internal"], check=True)
    # Even selecting NSF while saving mel must not open the absent external asset.
    subprocess.run(
        [*base, "--title", "features", "--save-mel", "--vocoder", "nsf-hifigan"],
        check=True,
    )
    saved = torch.load(
        args.output / "features.mel.pt", map_location="cpu", weights_only=False
    )
    assert len(saved) == 3 and all(torch.isfinite(item["mel"]).all() for item in saved)
    sample_rate = config["audio_sample_rate"]
    internal, rate = sf.read(args.output / "internal.wav")
    assert rate == sample_rate and internal.ndim == 1 and np.isfinite(internal).all()
    assert np.abs(internal).max() > 0
    assert (internal[: round(0.1 * rate)] == 0).all()
    # The actual mel frames determine the common post-processing timeline.
    lengths = [item["mel"].shape[1] * config["hop_size"] for item in saved]
    expected_length = round(segments[-1]["offset"] * rate) + lengths[-1]
    assert len(internal) == expected_length
    first_end = round(segments[0]["offset"] * rate) + lengths[0]
    second_start = round(segments[1]["offset"] * rate)
    assert second_start > first_end and (internal[first_end:second_start] == 0).all()
    assert round(segments[2]["offset"] * rate) < second_start + lengths[1]
    config["vocoder_ckpt"] = external_path
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    subprocess.run(
        [*base, "--title", "external", "--vocoder", "nsf-hifigan", "--vocoder-fp32"],
        check=True,
    )
    external, external_rate = sf.read(args.output / "external.wav")
    assert (
        external_rate == rate
        and external.shape == internal.shape
        and np.isfinite(external).all()
    )
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    slim_path = runtime / args.checkpoint.with_suffix(".infer.ckpt").name
    slim = torch.load(slim_path, map_location="cpu", weights_only=False)
    vocoder_keys = [
        key for key in original["state_dict"] if key.startswith("model.vocoder.")
    ]
    assert vocoder_keys and "optimizer_states" not in slim
    for key in vocoder_keys:
        torch.testing.assert_close(
            original["state_dict"][key], slim["state_dict"][key], rtol=0, atol=0
        )
    subprocess.run(
        [
            sys.executable,
            "scripts/infer.py",
            "all-in-one",
            str(project),
            "--exp",
            str(pathlib.Path("ckpt/dslight").resolve()),
            "--out",
            str(args.output),
            "--title",
            "legacy",
            "--seed",
            "42",
            "--steps",
            "20",
            "--vocoder-fp32",
        ],
        check=True,
    )
    legacy, legacy_rate = sf.read(args.output / "legacy.wav")
    assert (
        legacy_rate == rate
        and legacy.shape == internal.shape
        and np.isfinite(legacy).all()
    )
    from training.all_in_one_task import AllInOneTask
    from utils.hparams import hparams, set_hparams

    set_hparams(config=str(config_path), print_hparams=False)
    hparams["vocoder_ckpt"] = str(absent)
    hparams["val_with_vocoder"] = True
    validation_task = AllInOneTask()
    validation_task.load_finetune_ckpt(slim["state_dict"])
    validation_task.eval()
    assert validation_task.vocoder.internal
    validation_wave = validation_task.vocoder.spec2wav_torch(
        saved[0]["mel"], saved[0]["f0"]
    )
    assert (
        validation_wave.shape == (1, lengths[0])
        and torch.isfinite(validation_wave).all()
    )
    report = {
        "source": "2033001224",
        "seed": 42,
        "sampling_steps": 20,
        "segments": 3,
        "default_without_external": True,
        "save_mel_without_external": True,
        "nsf_cli": True,
        "legacy_cli": True,
        "training_validation_internal_without_external": True,
        "sample_rate": rate,
        "waveform_samples": len(internal),
        "offset_silence_crossfade": True,
        "vocoder_state_entries_preserved": len(vocoder_keys),
        "training_bytes": args.checkpoint.stat().st_size,
        "inference_bytes": slim_path.stat().st_size,
        "quality": "12-step integration checkpoint; quality evaluation belongs to M6",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
