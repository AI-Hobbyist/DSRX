"""Compare DSLight widths at fixed depth and receptive field before retraining."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.backbones.dslight import DSLightBackbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 160, 192])
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.widths + [args.frames, args.repeats, args.threads]) < 1:
        parser.error("Widths, frames, repeats and threads must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    backbone_args = config["backbone_args"].copy()
    torch.set_num_threads(args.threads)
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    for width in args.widths:
        torch.manual_seed(42)
        options = dict(backbone_args, num_channels=width)
        model = DSLightBackbone(config["audio_num_mel_bins"], 1, **options).eval()
        spec = torch.randn(1, 1, config["audio_num_mel_bins"], args.frames)
        condition = torch.randn(1, options["cond_channels"], args.frames)
        step = torch.tensor([400.0])
        with torch.inference_mode():
            prepared = model.prepare_condition(condition)
            torch.testing.assert_close(
                model(spec, step, prepared), model(spec, step, condition)
            )
            for _ in range(3):
                model(spec, step, prepared)
            durations = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                output = model(spec, step, prepared)
                durations.append(time.perf_counter() - start)
            if not torch.isfinite(output).all():
                raise RuntimeError(f"Nonfinite output at width {width}")
        parameters = sum(parameter.numel() for parameter in model.parameters())
        row = {
            "width": width,
            "parameters": parameters,
            "fp32_parameter_bytes": 4 * parameters,
            "median_prepared_forward_ms": 1000 * statistics.median(durations),
            "prepared_condition_bytes": prepared.numel() * prepared.element_size(),
        }
        results.append(row)
        candidate = {"base_config": [args.config.as_posix()], "backbone_args": options}
        (args.output / f"width_{width}.yaml").write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)
    report = {
        "scope": "Random-weight CPU backbone microbenchmark; not audio quality or end-to-end RTF",
        "checkpoint_compatibility": "Changed widths require retraining; do not load old weights non-strictly",
        "torch_version": torch.__version__,
        "threads": args.threads,
        "frames": args.frames,
        "repeats": args.repeats,
        "results": results,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
