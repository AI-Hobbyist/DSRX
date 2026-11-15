"""Measure the existing all-in-one CLI without changing its inference path."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf
import torch


@contextmanager
def timed_method(cls, timings, name, synchronize, memory, cuda):
    original = cls.run_inference

    def measured(instance, *args, **kwargs):
        synchronize()
        if name == "variance" and cuda:
            memory["model_loaded_allocated_mib"] = torch.cuda.memory_allocated() / 2**20
        started = time.perf_counter()
        result = original(instance, *args, **kwargs)
        synchronize()
        timings[name] = time.perf_counter() - started
        return result

    cls.run_inference = measured
    try:
        yield
    finally:
        cls.run_inference = original


def evaluate(ds_file, exp="dslight", ckpt=None, steps=None, out=None, device="auto"):
    # Resolve input before moving to the repo root, so callers can use relative
    # paths from any working directory. Model resources keep the CLI's semantics.
    ds_file = Path(ds_file).resolve(strict=True)
    output = (
        Path(out).resolve()
        if out is not None
        else ROOT
        / "outputs"
        / "benchmark"
        / ds_file.stem
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    from inference.ds_acoustic import DiffSingerAcousticInfer
    from inference.ds_variance import DiffSingerVarianceInfer
    from scripts.infer import main as infer_cli
    import utils
    from utils.hparams import hparams

    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported inference device: {device}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    cuda = device == "cuda"
    if cuda:
        torch.cuda.init()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline = None

    def synchronize():
        if cuda:
            torch.cuda.synchronize()

    memory = {}
    timings = {}
    checkpoint = {}
    original_load = utils.load_ckpt

    def record_checkpoint(model, *args, **kwargs):
        path = original_load(model, *args, **kwargs)
        checkpoint.update(
            path=str(Path(path).resolve()),
            file_mib=Path(path).stat().st_size / 2**20,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            device=str(next(model.parameters()).device),
        )
        return path

    arguments = [
        "all-in-one",
        str(ds_file),
        "--exp",
        exp,
        "--out",
        str(output),
        "--title",
        "result",
        "--vocoder",
        "mini-ddsp",
        "--device",
        device,
    ]
    if ckpt is not None:
        arguments.extend(["--ckpt", str(ckpt)])
    if steps is not None:
        arguments.extend(["--steps", str(steps)])

    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    utils.load_ckpt = record_checkpoint
    try:
        with timed_method(
            DiffSingerVarianceInfer, timings, "variance", synchronize, memory, cuda
        ), timed_method(
            DiffSingerAcousticInfer,
            timings,
            "acoustic_and_vocoder",
            synchronize,
            memory,
            cuda,
        ):
            started = time.perf_counter()
            infer_cli.main(args=arguments, standalone_mode=False)
            synchronize()
            total_seconds = time.perf_counter() - started
        if cuda:
            peak = torch.cuda.max_memory_allocated() / 2**20
            memory.update(
                baseline_allocated_mib=baseline / 2**20,
                peak_allocated_mib=peak,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                incremental_peak_allocated_mib=peak - baseline / 2**20,
                peak_above_loaded_model_mib=max(
                    0.0, peak - memory["model_loaded_allocated_mib"]
                ),
            )
        waveform = output / "result.wav"
        info = sf.info(str(waveform))
        duration = info.frames / info.samplerate
        if duration <= 0:
            raise ValueError("Inference produced empty audio")
        predicted = json.loads(
            (output / "result_variance.ds").read_text(encoding="utf-8")
        )
        # Output WAV can contain a leading offset and inter-segment silence.
        # Also report a duration without those gaps to avoid inflated speedups.
        segment_seconds = sum(
            sum(float(value) for value in segment["ph_dur"].split())
            for segment in predicted
        )
        pipeline_seconds = sum(timings.values())
        report = {
            "input_ds": str(ds_file),
            "checkpoint": checkpoint,
            "device": torch.cuda.get_device_name() if cuda else "cpu",
            "device_type": device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "sampling_steps": hparams["sampling_steps"],
            "seed": 1234,
            "segment_count": len(predicted),
            "output_wav": str(waveform),
            "audio_seconds": duration,
            "sample_rate": info.samplerate,
            "predicted_segment_seconds_without_offset_gaps": segment_seconds,
            "timings_seconds": {
                **timings,
                "pipeline": pipeline_seconds,
                "loading_and_setup": total_seconds - pipeline_seconds,
                "total_with_loading": total_seconds,
            },
            "rtf": pipeline_seconds / duration,
            "rtf_with_loading": total_seconds / duration,
            "speedup_realtime": duration / pipeline_seconds,
            "rtf_without_offset_gaps": (
                pipeline_seconds / segment_seconds if segment_seconds > 0 else None
            ),
            "cuda_memory_mib": memory if cuda else None,
            "measurement_notes": [
                "Single cold inference; CUDA synchronized at timing boundaries.",
                "Pipeline includes input preprocessing and output writes; excludes model loading.",
                "Total excludes Python startup/imports; includes model loading and setup.",
                "RTF = seconds / output WAV duration (including timeline silence).",
                "CUDA peaks include model loading; allocated is tensor memory, reserved includes allocator cache.",
                "CUDA driver/context, other processes and host RAM are not included in allocator measurements.",
                "No reference audio: these are performance metrics, not audio-quality scores.",
            ],
        }
        report_path = output / "metrics.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        utils.load_ckpt = original_load
        os.chdir(previous_cwd)
    print("\n=== Inference benchmark ===")
    print(f"Device: {device} ({report['device']})")
    print(
        f"Audio: {duration:.3f}s; pipeline: {pipeline_seconds:.3f}s; total: {total_seconds:.3f}s"
    )
    print(
        f"RTF: {report['rtf']:.4f}; speed: {report['speedup_realtime']:.2f}x realtime"
    )
    print(f"RTF without timeline gaps: {report['rtf_without_offset_gaps']}")
    if cuda:
        print(f"CUDA peak allocated: {memory['peak_allocated_mib']:.1f} MiB")
        print(f"CUDA peak reserved: {memory['peak_reserved_mib']:.1f} MiB")
        print(
            f"Loaded model allocation: {memory['model_loaded_allocated_mib']:.1f} MiB"
        )
        print(
            f"Peak above loaded model: {memory['peak_above_loaded_model_mib']:.1f} MiB"
        )
    print(f"Report: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ds_file", type=Path, help="Input DS project")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--exp", default="dslight", help="Model folder under ckpt (default: dslight)"
    )
    parser.add_argument(
        "--ckpt", type=int, help="Checkpoint training step; default: latest"
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override sampling steps; default: model configuration",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory; default: timestamped outputs/benchmark",
    )
    args = parser.parse_args()
    evaluate(
        args.ds_file,
        exp=args.exp,
        ckpt=args.ckpt,
        steps=args.steps,
        out=args.out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
