"""Verify a real waveform deployment in an isolated, source-free ORT process."""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import torch
import yaml

from inference.ds_acoustic import DiffSingerAcousticInfer
from inference.ds_variance import DiffSingerVarianceInfer
from modules.fastspeech.tts_modules import mel2ph_to_dur
from modules.toplevel import DiffSingerAllInOne
from modules.vocoders.all_in_one import build_all_in_one_vocoder
from utils import load_ckpt
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary

ISOLATED_PROBE = """
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
manifest = json.loads(Path("manifest.json").read_text())
def session(path):
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
def inputs(model, values):
    types = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
    return {item.name: np.asarray(values[item.name], dtype=types[item.type]) for item in model.get_inputs()}
wave = session(manifest["waveform"])
mel = session(manifest["mel_only"])
cases = json.loads(Path("cases.json").read_text())
for index, case in enumerate(cases):
    values = inputs(wave, case)
    result = dict(zip([item.name for item in wave.get_outputs()], wave.run(None, values)))
    np.savez("result_%d.npz" % index, **result)
    if case["depth"] == 0:
        mel_result = dict(zip([item.name for item in mel.get_outputs()], mel.run(None, inputs(mel, case))))
        np.testing.assert_allclose(mel_result["mel"], result["mel"], rtol=2e-4, atol=2e-5)

config_path = Path(manifest["variance_config"])
# The fixture supplies the same real score/alignment as native validation.
variance = json.loads(Path("variance_inputs.json").read_text())
graphs = json.loads(Path("variance_graphs.json").read_text())
linguistic = session(config_path.parent / graphs["linguistic"])
ling_out = dict(zip([item.name for item in linguistic.get_outputs()], linguistic.run(None, inputs(linguistic, variance))))
variance.update({key: value.tolist() for key, value in ling_out.items()})
executed = {}
for name in ("dur", "pitch", "variance"):
    model = session(config_path.parent / graphs[name])
    values = dict(variance)
    if name == "variance":
        values["retake"] = np.ones((1, len(variance["pitch"][0]), 3), dtype=bool).tolist()
    result = model.run(None, inputs(model, values))
    assert all(np.isfinite(value).all() for value in result)
    executed[name] = {item.name: list(value.shape) for item, value in zip(model.get_outputs(), result)}
Path("isolated_report.json").write_text(json.dumps({"cases": len(cases), "variance": executed}))
print("Isolated ORT waveform, mel and variance graphs passed")
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/mini_ddsp_m5/deploy"),
    )
    parser.add_argument(
        "--runtime",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/mini_ddsp_m4/runtime"),
    )
    parser.add_argument(
        "--project",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/mini_ddsp_m4/features_variance.ds"),
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("artifacts/mini_ddsp_m5")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    set_hparams(config=str(args.runtime / "config.yaml"), print_hparams=False)
    hparams["work_dir"] = str(args.runtime)
    hparams["infer"] = False
    hparams["inference_optimization"] = False
    model = DiffSingerAllInOne(
        len(load_phoneme_dictionary()), hparams["audio_num_mel_bins"]
    ).eval()
    load_ckpt(model, args.runtime, strict=True, prefer_inference=True)
    acoustic_infer = DiffSingerAcousticInfer(
        device="cpu", model=model.acoustic, load_vocoder=False
    )
    project = json.loads(args.project.read_text(encoding="utf-8"))[0]
    batch = acoustic_infer.preprocess_input(project)
    durations = mel2ph_to_dur(batch["mel2ph"], batch["tokens"].shape[1])
    real = {
        "tokens": batch["tokens"].tolist(),
        "durations": durations.tolist(),
        "f0": batch["f0"].tolist(),
        "depth": 0.6,
        "steps": 20,
    }
    for name in ("breathiness", "voicing", "tension"):
        real[name] = batch[name].tolist()
    frames = len(real["f0"][0])
    real["gender"], real["velocity"] = [[0.0] * frames], [[1.0] * frames]
    cases = []
    for count in (1, 9, 11, 257, 1025):
        tokens = real["tokens"][0][: min(3, count)]
        lengths = [count // len(tokens)] * len(tokens)
        lengths[-1] += count - sum(lengths)
        case = {
            "tokens": [tokens],
            "durations": [lengths],
            "f0": [np.linspace(150, 400, count, dtype=np.float32).tolist()],
            "depth": 0.0 if count in (1, 9) else 0.6,
            "steps": 3,
            "gender": [[0.0] * count],
            "velocity": [[1.0] * count],
        }
        for name in ("breathiness", "voicing", "tension"):
            case[name] = [[float(batch[name][0].median())] * count]
        cases.append(case)
    cases.append(real)
    variance_infer = DiffSingerVarianceInfer(
        device="cpu", model=model.variance, predictions={"dur", "pitch"}
    )
    variance_batch = variance_infer.preprocess_input(project, load_dur=True)
    variance_inputs = {
        "tokens": batch["tokens"].tolist(),
        "word_div": [list(map(int, project["ph_num"].split()))],
        "word_dur": variance_batch["word_dur"].tolist(),
        "ph_midi": variance_batch["midi"].tolist(),
        "ph_dur": durations.tolist(),
        "note_midi": variance_batch["note_midi"].tolist(),
        "note_dur": variance_batch["note_dur"].tolist(),
        "note_rest": variance_batch["note_rest"].tolist(),
        "pitch": variance_batch["base_pitch"].tolist(),
        "expr": [[1.0] * frames],
        "retake": [[True] * frames],
        "steps": 3,
        **{
            name: batch[name].tolist() for name in ("breathiness", "voicing", "tension")
        },
    }
    manifest = json.loads((args.deploy / "manifest.json").read_text())
    graphs = yaml.safe_load(
        (args.deploy / manifest["variance_config"]).read_text(encoding="utf-8")
    )
    for path in args.deploy.rglob("*.onnx"):
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        assert all(not value.external_data for value in graph.graph.initializer)
    mel_graph = onnx.load(args.deploy / manifest["mel_only"])
    assert not any(node.name.startswith("mini_ddsp.") for node in mel_graph.graph.node)
    reports = []
    with tempfile.TemporaryDirectory(prefix="dsrx-ort-isolated-") as directory:
        isolated = pathlib.Path(directory)
        shutil.copytree(args.deploy, isolated, dirs_exist_ok=True)
        (isolated / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
        (isolated / "variance_inputs.json").write_text(
            json.dumps(variance_inputs), encoding="utf-8"
        )
        (isolated / "variance_graphs.json").write_text(
            json.dumps(graphs), encoding="utf-8"
        )
        subprocess.run(
            [sys.executable, "-I", "-c", ISOLATED_PROBE], cwd=isolated, check=True
        )
        for index, case in enumerate(cases):
            with np.load(isolated / f"result_{index}.npz") as result:
                mel, waveform, lengths = (
                    result["mel"],
                    result["waveform"],
                    result["waveform_lengths"],
                )
            f0 = torch.tensor(case["f0"], dtype=torch.float32)
            with torch.no_grad():
                expected = model.vocoder(torch.from_numpy(mel), f0).numpy()
            np.testing.assert_allclose(waveform, expected, rtol=2e-4, atol=2e-5)
            np.testing.assert_array_equal(lengths, [f0.shape[1] * hparams["hop_size"]])
            report = {
                "frames": f0.shape[1],
                "depth": case["depth"],
                "waveform_max_abs": float(np.abs(waveform - expected).max()),
            }
            if case["depth"] == 0:
                hparams["T_start_infer"] = 1.0
                tokens = torch.tensor(case["tokens"])
                mel2ph = torch.repeat_interleave(
                    torch.arange(1, tokens.shape[1] + 1),
                    torch.tensor(case["durations"][0]),
                )[None]
                with torch.no_grad():
                    eager = model.acoustic(
                        tokens,
                        mel2ph,
                        f0,
                        key_shift=torch.tensor(case["gender"]),
                        speed=torch.tensor(case["velocity"]),
                        spk_embed_id=torch.zeros(1, dtype=torch.long),
                        languages=torch.zeros_like(tokens),
                        **{
                            name: torch.tensor(case[name])
                            for name in ("breathiness", "voicing", "tension")
                        },
                        infer=True,
                    ).diff_out.numpy()
                np.testing.assert_allclose(mel, eager, rtol=2e-4, atol=2e-5)
                report["deterministic_acoustic_max_abs"] = float(
                    np.abs(mel - eager).max()
                )
            reports.append(report)
            if index == len(cases) - 1:
                nsf = build_all_in_one_vocoder(model, "nsf-hifigan")
                rendered = nsf.spec2wav_torch(torch.from_numpy(mel), f0)
                assert (
                    rendered.shape == waveform.shape and torch.isfinite(rendered).all()
                )
        independent = json.loads((isolated / "isolated_report.json").read_text())
    report = {
        "opset": 15,
        "source_free_ort": True,
        "cases": reports,
        "variance_graphs_executed": independent["variance"],
        "mel_only_excludes_ddsp": True,
        "nsf_from_exported_mel": True,
        "comparison": "waveform vs eager vocoder on graph mel; deterministic depth0 also checks acoustic; stochastic noise is runtime-specific",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
