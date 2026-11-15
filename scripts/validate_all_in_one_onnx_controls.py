"""Source-free functional checks for complete all-in-one ONNX deployment."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnxruntime as ort


def fixture(frames, mode, curves):
    phones = 1 if frames == 1 else 4
    divisions = [1] if phones == 1 else [2, 2]
    word_dur = [frames] if phones == 1 else [frames // 2, frames - frames // 2]
    durations = [frames // phones] * phones
    durations[-1] += frames - sum(durations)
    positions = np.arange(frames)[None, :]
    pitch_retake = np.full((1, frames), mode == "automatic")
    retake = np.full((1, frames, len(curves)), mode == "automatic")
    if mode == "mixed":
        pitch_retake = positions % 2 == 0
        for index in range(len(curves)):
            retake[:, :, index] = (positions + index) % 3 == 0
    result = {
        "tokens": np.ones((1, phones), np.int64),
        "word_div": np.array([divisions], np.int64),
        "word_dur": np.array([word_dur], np.int64),
        "ph_midi": np.full((1, phones), 60, np.int64),
        "note_midi": np.full((1, len(word_dur)), 60, np.float32),
        "note_rest": np.zeros((1, len(word_dur)), bool),
        "note_dur": np.array([word_dur], np.int64),
        "manual_durations": np.array([durations], np.int64),
        "use_manual_durations": np.array(mode == "manual"),
        "manual_f0": np.full((1, frames), 220, np.float32),
        "expr": np.ones((1, frames), np.float32),
        "pitch_retake": pitch_retake,
        "variance_retake": retake,
        "gender": np.zeros((1, frames), np.float32),
        "velocity": np.ones((1, frames), np.float32),
        "depth": np.array(0.6 if frames == 11 else 0.0, np.float32),
        "pitch_steps": np.array(2, np.int64),
        "variance_steps": np.array(2, np.int64),
        "acoustic_steps": np.array(2, np.int64),
    }
    for index, curve in enumerate(curves):
        result[f"manual_{curve}"] = np.full((1, frames), -40 + index * 10, np.float32)
    result["manual_f0"][0, -1] = 0.0
    return result


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(directory / manifest["full"]), options)
    mel_session = ort.InferenceSession(
        str(directory / manifest["full_mel_only"]), options
    )
    names = {value.name for value in session.get_inputs()}
    curves = [
        name
        for name in ("energy", "breathiness", "voicing", "tension")
        if "manual_" + name in names
    ]
    reports = []
    for frames in (1, 9, 10, 11, 257, 1025):
        for mode in ("automatic", "manual", "mixed"):
            values = fixture(frames, mode, curves)
            feed = {value.name: values[value.name] for value in session.get_inputs()}
            result = dict(
                zip(
                    [value.name for value in session.get_outputs()],
                    session.run(None, feed),
                )
            )
            assert all(np.isfinite(value).all() for value in result.values())
            assert result["waveform"].shape == (1, frames * manifest["hop_size"])
            np.testing.assert_array_equal(
                result["waveform_lengths"], [frames * manifest["hop_size"]]
            )
            assert result["durations"].sum() == frames
            manual_pitch = ~values["pitch_retake"]
            np.testing.assert_array_equal(
                result["f0"][manual_pitch], values["manual_f0"][manual_pitch]
            )
            for index, curve in enumerate(curves):
                mask = ~values["variance_retake"][:, :, index]
                np.testing.assert_array_equal(
                    result[curve][mask], values["manual_" + curve][mask]
                )
            if mode == "manual":
                np.testing.assert_array_equal(
                    result["durations"], values["manual_durations"]
                )
                if values["depth"] == 0:
                    mel_feed = {
                        value.name: values[value.name]
                        for value in mel_session.get_inputs()
                    }
                    mel = dict(
                        zip(
                            [value.name for value in mel_session.get_outputs()],
                            mel_session.run(None, mel_feed),
                        )
                    )
                    np.testing.assert_allclose(
                        result["mel"], mel["mel"], rtol=2e-4, atol=2e-5
                    )
            reports.append({"frames": frames, "mode": mode})
            print(f"PASS full ONNX: {frames} frames, {mode}", flush=True)
    return {
        "source_free": True,
        "cases": reports,
        "curves": curves,
        "quality": "not evaluated",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.worker:
        report = verify(args.deploy)
        (args.deploy / "controls-report.json").write_text(json.dumps(report, indent=2))
        return
    manifest = json.loads((args.deploy / "manifest.json").read_text())
    with tempfile.TemporaryDirectory() as temporary:
        isolated = Path(temporary)
        # Only the two complete views are available; no source, split models,
        # checkpoint, external vocoder or external noise resource is copied.
        for key in ("full", "full_mel_only"):
            shutil.copyfile(args.deploy / manifest[key], isolated / manifest[key])
        (isolated / "manifest.json").write_text(json.dumps(manifest))
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                "--deploy",
                str(isolated),
                "--worker",
            ],
            cwd=isolated,
            check=True,
        )
        report = (isolated / "controls-report.json").read_text()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report)


if __name__ == "__main__":
    main()
