"""Evaluate acoustic shallow Reflow on the checkpoint's held-out raw recordings."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.toplevel import DiffSingerAcoustic
from preprocessing.acoustic_binarizer import AcousticBinarizer
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def load_acoustic(config_path, checkpoint_path, device):
    set_hparams(config=str(config_path), print_hparams=False)
    # Use the dictionary shipped with this checkpoint, preserving its token IDs.
    hparams["dictionaries"] = {
        language: str(config_path.parent / f"dictionary-{language}.txt")
        for language in hparams["dictionaries"]
    }
    hparams["infer"] = False
    model = DiffSingerAcoustic(
        len(load_phoneme_dictionary()), hparams["audio_num_mel_bins"]
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    prefix = "model.acoustic." if checkpoint["category"] == "all_in_one" else "model."
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def validation_items(device):
    binarizer = AcousticBinarizer()
    binarizer.device = torch.device(device)
    for dataset_index, dataset in enumerate(hparams["datasets"]):
        metadata = binarizer.load_meta_data(
            Path(dataset["raw_data_dir"]),
            dataset_index,
            dataset["speaker"],
            dataset["language"],
        )
        prefixes = tuple(str(value) for value in dataset.get("test_prefixes", []))
        for name, entry in sorted(metadata.items()):
            if prefixes and name.split(":", 1)[1].startswith(prefixes):
                yield name, binarizer.process_item(
                    name, entry, hparams["binarization_args"]
                )


def acoustic_inputs(item, device):
    inputs = {}
    for key in (
        "tokens",
        "mel2ph",
        "f0",
        "languages",
        "energy",
        "breathiness",
        "voicing",
        "tension",
    ):
        if key in item:
            inputs["txt_tokens" if key == "tokens" else key] = torch.as_tensor(
                item[key], device=device
            ).unsqueeze(0)
    for key in ("key_shift", "speed"):
        if key in item:
            inputs[key] = torch.tensor([[item[key]]], device=device)
    if "artifact_level" in item:
        inputs["artifact_level"] = torch.tensor([item["artifact_level"]], device=device)
    if hparams.get("use_spk_id"):
        inputs["spk_embed_id"] = torch.tensor([item["spk_id"]], device=device)
    return inputs


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 20])
    parser.add_argument("--starts", type=float, nargs="+", default=[0.4, 0.7, 1.0])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.limit < 1
        or min(args.steps) < 1
        or any(not 0 <= x <= 1 for x in args.starts)
    ):
        parser.error("Positive limit/steps and starts in [0,1] are required")
    torch.set_num_threads(1)
    model = load_acoustic(args.config, args.checkpoint, args.device)
    if model.diffusion_type != "reflow" or not model.use_shallow_diffusion:
        parser.error("This evaluation requires a shallow Reflow acoustic checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (name, item) in enumerate(validation_items(args.device)):
        if item is None:
            raise RuntimeError(f"Feature extraction failed for held-out item {name}")
        inputs = acoustic_inputs(item, args.device)
        target = torch.as_tensor(item["mel"], device=args.device).unsqueeze(0)
        # Reusable real-data features; no need to re-extract for quantization comparisons.
        torch.save(
            {
                "name": name,
                "inputs": {k: v.cpu() for k, v in inputs.items()},
                "target": target.cpu(),
                "source_audio": item["wav_fn"],
            },
            args.output / f"sample_{index}.pt",
        )
        nfe = [0]

        def count_call(*unused):
            nfe[0] += 1

        handle = model.diffusion.velocity_fn.register_forward_hook(count_call)
        try:
            with torch.inference_mode():
                # Exclude one-time kernel initialization from measured inference.
                hparams.update(
                    sampling_algorithm="euler", sampling_steps=1, T_start_infer=0.4
                )
                model(**inputs, infer=True)
                for start in args.starts:
                    for steps in (args.steps if start < 1 else [0]):
                        hparams.update(sampling_steps=steps, T_start_infer=start)
                        torch.manual_seed(42 + index)
                        nfe[0] = 0
                        synchronize(args.device)
                        begin = time.perf_counter()
                        result = model(**inputs, infer=True)
                        if nfe[0] != steps:
                            raise RuntimeError(
                                f"Unexpected Euler NFE: {nfe[0]} != {steps}"
                            )
                        synchronize(args.device)
                        seconds = time.perf_counter() - begin
                        if not torch.isfinite(result.diff_out).all():
                            raise RuntimeError(f"Nonfinite inference output: {name}")
                        row = {
                            "name": name,
                            "start": start,
                            "steps": steps,
                            "nfe": nfe[0],
                            "aux_mel_mae": (result.aux_out - target)
                            .abs()
                            .mean()
                            .item(),
                            "final_mel_mae": (result.diff_out - target)
                            .abs()
                            .mean()
                            .item(),
                            "inference_seconds": seconds,
                            "acoustic_rtf": seconds / item["seconds"],
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
        finally:
            handle.remove()
        if index + 1 >= args.limit:
            break
    if not rows:
        raise RuntimeError(
            "No checkpoint-held-out recordings matched the supplied dataset"
        )
    report = {
        "scope": "Held-out acoustic evaluation with reference duration, pitch and variance controls",
        "quality_limit": "Mel MAE is a reconstruction proxy, not a listening or end-to-end SVS verdict",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device": args.device,
        "torch_version": torch.__version__,
        "results": rows,
    }
    (args.output / "sampling.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
