"""Measure existing Mini-DDSP/NSF vocoders on the same held-out mel and F0."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.vocoders.all_in_one import build_all_in_one_vocoder
from modules.vocoders.mini_ddsp import MiniDDSP
from utils import load_ckpt
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDataset


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--nsf", type=Path, required=True)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    set_hparams(config=str(args.runtime / "config.yaml"), print_hparams=False)
    hparams["vocoder_ckpt"] = str(args.nsf)
    hparams["inference_vocoder_nsf_selective_fp16"] = False
    device = torch.device("cuda")
    vocoder = MiniDDSP(hparams).eval().to(device)
    load_ckpt(
        vocoder,
        args.checkpoint,
        prefix_in_ckpt="model.vocoder",
        strict=True,
        device=device,
    )
    model = torch.nn.Module()
    model.add_module("vocoder", vocoder)
    dataset = IndexedDataset(args.binary / "vocoder", "valid")
    results = {}
    for backend in ("mini-ddsp", "nsf-hifigan"):
        adapter = build_all_in_one_vocoder(model, backend)
        measurements = []
        for index in range(len(dataset)):
            item = dataset[index]
            mel, f0 = item["mel"][None].to(device), item["f0"][None].to(device)
            for _ in range(2):
                adapter.spec2wav_torch(mel, f0=f0)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                waveform = adapter.spec2wav_torch(mel, f0=f0)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
            assert waveform.shape == (1, mel.shape[1] * hparams["hop_size"])
            assert torch.isfinite(waveform).all()
            duration = waveform.shape[1] / hparams["audio_sample_rate"]
            latency = statistics.median(timings)
            measurements.append(
                {
                    "sample_id": item["sample_id"].decode(),
                    "seconds": duration,
                    "median_latency_seconds": latency,
                    "rtf": latency / duration,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "extra_peak_cuda_bytes": torch.cuda.max_memory_allocated()
                    - baseline,
                }
            )
            del waveform
        results[backend] = {
            "items": measurements,
            "aggregate_rtf": sum(
                item["median_latency_seconds"] for item in measurements
            )
            / sum(item["seconds"] for item in measurements),
        }
        del adapter
    dataset.dset.close()
    dataset.dset = None
    manifest = json.loads((args.deploy / "manifest.json").read_text())
    report = {
        "seed": 42,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": "FP32; Mini-DDSP phase accumulation FP64; NSF selective FP16 disabled",
        "checkpoint_sha256": sha256(args.checkpoint),
        "config_sha256": sha256(args.runtime / "config.yaml"),
        "checkpoint": str(args.checkpoint),
        "binary": str(args.binary),
        "warmup_calls_per_item": 2,
        "measured_calls_per_item": 3,
        "sample_rate": hparams["audio_sample_rate"],
        "hop_size": hparams["hop_size"],
        "backends": results,
        "deployment_bytes": {
            key: (args.deploy / manifest[key]).stat().st_size
            for key in ("full", "full_mel_only", "waveform")
        },
        "quality_and_convergence": "SKIPPED: user decision; this small-step model only verifies functionality and runtime cost",
        "spectral_and_f0_quality_metrics": "SKIPPED: require quality-trained weights; not inferred from timing or finite outputs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
