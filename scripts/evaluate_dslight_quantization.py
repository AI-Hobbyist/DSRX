"""Compare opt-in INT8 weight storage against FP32 on cached real-data features."""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_dslight_sampling import load_acoustic, synchronize
from modules.backbones.dslight_quantization import (
    load_weight_only_package,
    quantize_dslight_pointwise,
)
from utils.hparams import hparams


class UnpreparedBackbone(torch.nn.Module):
    """Exercise the ordinary condition path without exposing the preparation hook."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, spec, step, condition):
        return self.backbone(spec, step, condition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--render-audio", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("Steps and repeats must be positive")
    torch.set_num_threads(1)
    baseline = load_acoustic(args.config, args.checkpoint, args.device)
    uncached = copy.deepcopy(baseline)
    uncached.diffusion.velocity_fn = UnpreparedBackbone(uncached.diffusion.velocity_fn)
    quantized = copy.deepcopy(baseline)
    layers = quantize_dslight_pointwise(quantized)
    if not layers:
        raise RuntimeError("No DSLight projections were selected")
    args.output.mkdir(parents=True, exist_ok=True)
    float_path = args.output / "acoustic-fp32.pt"
    quantized_path = args.output / "acoustic-int8-weight-only.pt"
    torch.save(
        {key: value.cpu() for key, value in baseline.state_dict().items()}, float_path
    )
    package = {
        "format": "dslight-int8-weight-only-v1",
        "layers": layers,
        "state_dict": {
            key: value.cpu() for key, value in quantized.state_dict().items()
        },
    }
    torch.save(package, quantized_path)
    restored = load_weight_only_package(
        copy.deepcopy(baseline),
        torch.load(quantized_path, map_location=args.device, weights_only=True),
    )
    hparams.update(
        sampling_algorithm="euler", sampling_steps=args.steps, T_start_infer=0.4
    )
    rows = []
    vocoder = None
    if args.render_audio:
        import soundfile as sf
        from modules.vocoders.nsf_hifigan import NsfHifiGAN

        vocoder = NsfHifiGAN()
        vocoder.to_device(args.device)
    for path in sorted(args.features.glob("sample_*.pt")):
        sample = torch.load(path, map_location=args.device, weights_only=True)
        outputs = {}
        durations = {}
        with torch.inference_mode():
            for label, model in [
                ("fp32_uncached", uncached),
                ("fp32", baseline),
                ("int8_weight_only", quantized),
            ]:
                torch.manual_seed(42)
                model(**sample["inputs"], infer=True)
                timings = []
                for _ in range(args.repeats):
                    torch.manual_seed(42)
                    synchronize(args.device)
                    start = time.perf_counter()
                    result = model(**sample["inputs"], infer=True).diff_out
                    synchronize(args.device)
                    timings.append(time.perf_counter() - start)
                if not torch.isfinite(result).all():
                    raise RuntimeError(f"Nonfinite {label} output")
                outputs[label] = result
                durations[label] = statistics.median(timings)
            torch.manual_seed(42)
            reloaded = restored(**sample["inputs"], infer=True).diff_out
            torch.testing.assert_close(
                reloaded, outputs["int8_weight_only"], rtol=0, atol=0
            )
            torch.testing.assert_close(outputs["fp32_uncached"], outputs["fp32"])
        row = {
            "name": sample["name"],
            "median_seconds": durations,
            "fp32_target_mel_mae": (outputs["fp32"] - sample["target"])
            .abs()
            .mean()
            .item(),
            "quantized_target_mel_mae": (outputs["int8_weight_only"] - sample["target"])
            .abs()
            .mean()
            .item(),
            "quantized_vs_fp32_mel_mae": (outputs["int8_weight_only"] - outputs["fp32"])
            .abs()
            .mean()
            .item(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if vocoder is not None:
            with torch.inference_mode():
                hparams["T_start_infer"] = 1.0
                outputs["auxiliary"] = baseline(**sample["inputs"], infer=True).aux_out
                hparams.update(T_start_infer=0.4, sampling_steps=4)
                torch.manual_seed(42)
                outputs["fp32_4steps"] = baseline(
                    **sample["inputs"], infer=True
                ).diff_out
                hparams["sampling_steps"] = args.steps
                outputs["reference_mel"] = sample["target"]
                for label in (
                    "fp32",
                    "int8_weight_only",
                    "auxiliary",
                    "fp32_4steps",
                    "reference_mel",
                ):
                    torch.manual_seed(42)
                    audio = vocoder.spec2wav_torch(
                        outputs[label], f0=sample["inputs"]["f0"]
                    )
                    if not torch.isfinite(audio).all():
                        raise RuntimeError(f"Nonfinite waveform: {label}")
                    sf.write(
                        args.output / f"{path.stem}_{label}.wav",
                        audio.cpu().numpy(),
                        hparams["audio_sample_rate"],
                    )
    if not rows:
        raise RuntimeError("No real-data cached features found")
    report = {
        "scope": "INT8 weight storage with float convolution, not integer-kernel acceleration",
        "device": args.device,
        "torch_version": torch.__version__,
        "steps": args.steps,
        "converted_layers": layers,
        "fp32_bytes": float_path.stat().st_size,
        "quantized_bytes": quantized_path.stat().st_size,
        "results": rows,
        "quality_status": "Real held-out mel proxy measured; listening MANUAL/PENDING",
    }
    (args.output / "quantization.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Package bytes: {report['fp32_bytes']} -> {report['quantized_bytes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
