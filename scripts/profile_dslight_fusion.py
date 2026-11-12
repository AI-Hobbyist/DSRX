"""Measure explicit gate/projection compilation on cached real acoustic inputs."""

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
from modules.backbones.dslight_fusion import enable_dslight_fusion
from utils.hparams import hparams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline = load_acoustic(args.config, args.checkpoint, args.device)
    fused = copy.deepcopy(baseline)
    converted = enable_dslight_fusion(fused)
    hparams.update(sampling_algorithm="euler", sampling_steps=20, T_start_infer=0.4)
    rows = []
    for path in sorted(args.features.glob("sample_*.pt")):
        sample = torch.load(path, map_location=args.device, weights_only=True)
        results = {}
        with torch.inference_mode():
            for name, model in [("eager", baseline), ("compiled_gate", fused)]:
                torch.manual_seed(42)
                start = time.perf_counter()
                model(**sample["inputs"], infer=True)
                synchronize(args.device)
                warmup_seconds = time.perf_counter() - start
                timings = []
                for _ in range(3):
                    torch.manual_seed(42)
                    synchronize(args.device)
                    start = time.perf_counter()
                    output = model(**sample["inputs"], infer=True).diff_out
                    synchronize(args.device)
                    timings.append(time.perf_counter() - start)
                results[name] = output
                row = {
                    "name": sample["name"],
                    "variant": name,
                    "warmup_seconds": warmup_seconds,
                    "median_seconds": statistics.median(timings),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            torch.testing.assert_close(
                results["eager"], results["compiled_gate"], atol=1e-4, rtol=1e-4
            )
            shorter = dict(sample["inputs"])
            frames = sample["target"].shape[1] // 2
            for key in ("mel2ph", "f0", "energy", "breathiness", "voicing", "tension"):
                if key in shorter:
                    shorter[key] = shorter[key][:, :frames]
            torch.manual_seed(42)
            expected = baseline(**shorter, infer=True).diff_out
            torch.manual_seed(42)
            actual = fused(**shorter, infer=True).diff_out
            torch.testing.assert_close(expected, actual, atol=1e-4, rtol=1e-4)
            backbone = fused.diffusion.velocity_fn
            spec = torch.randn(
                1, 1, hparams["audio_num_mel_bins"], 17, device=args.device
            )
            step = torch.tensor([400.0], device=args.device)
            condition = torch.randn(1, hparams["hidden_size"], 17, device=args.device)
            traced = torch.jit.trace(backbone, (spec, step, condition))
            torch.testing.assert_close(
                backbone(spec, step, condition),
                traced(spec, step, condition),
                atol=1e-4,
                rtol=1e-4,
            )
    if not rows:
        raise RuntimeError("No cached real features found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "device": args.device,
                "converted": converted,
                "dynamic_length_checked": True,
                "trace_fallback_checked": True,
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
