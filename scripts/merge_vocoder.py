"""Embed independent Mini-DDSP or NSF-HiFiGAN weights in an AIO experiment."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hparams import set_hparams
from utils.mini_ddsp_checkpoint import merge_vocoder_weights


def merge_nsf_weights(aio_checkpoint, vocoder_checkpoint, config, metadata):
    from modules.vocoders.all_in_one import build_embedded_nsf, validate_nsf_contract

    if aio_checkpoint.get("category") != "all_in_one":
        raise ValueError("The destination checkpoint must be all_in_one")
    validate_nsf_contract(config, metadata)
    generator = build_embedded_nsf(metadata)
    generator.load_state_dict(vocoder_checkpoint["generator"], strict=True)
    prefix = "model.vocoder."
    state = {
        name: tensor
        for name, tensor in aio_checkpoint["state_dict"].items()
        if not name.startswith(prefix)
    }
    state.update(
        {prefix + name: tensor for name, tensor in generator.state_dict().items()}
    )
    return {
        "category": "all_in_one",
        "state_dict": state,
        "global_step": 0,
        "vocoder_merge": {
            "aio_step": aio_checkpoint.get("global_step"),
            "backend": "nsf-hifigan",
        },
    }


def merge_experiment(aio_path, vocoder_path, output_dir, config_path=None):
    aio_path = Path(aio_path).resolve()
    vocoder_path = Path(vocoder_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output experiment already exists: {output_dir}")
    config_path = Path(config_path) if config_path else aio_path.parent / "config.yaml"
    config = set_hparams(
        config=str(config_path), print_hparams=False, global_hparams=False
    )
    if config.get("lora", {}).get("enabled", False):
        raise ValueError("Merge requires a full AIO checkpoint with LoRA disabled")
    # Input checkpoints are trusted local training artifacts, including Lightning metadata.
    aio_checkpoint = torch.load(aio_path, map_location="cpu", weights_only=False)
    vocoder_checkpoint = torch.load(
        vocoder_path, map_location="cpu", weights_only=False
    )
    aio_settings = config.setdefault("all_in_one", {})
    aio_settings["enabled"] = True
    mini = aio_settings.setdefault("mini_ddsp", {})
    if vocoder_checkpoint.get("category") == "vocoder":
        replacing_nsf = aio_settings.pop("embedded_nsf", None) is not None
        has_vocoder = any(
            key.startswith("model.vocoder.") for key in aio_checkpoint["state_dict"]
        )
        if replacing_nsf:
            aio_checkpoint = {
                **aio_checkpoint,
                "state_dict": {
                    key: value
                    for key, value in aio_checkpoint["state_dict"].items()
                    if not key.startswith("model.vocoder.")
                },
            }
        if replacing_nsf or not has_vocoder:
            mini.update(vocoder_checkpoint["mini_ddsp_contract"]["model"])
        mini["enabled"] = True
        merged = merge_vocoder_weights(aio_checkpoint, vocoder_checkpoint, config)
    elif "generator" in vocoder_checkpoint:
        metadata = json.loads(
            vocoder_path.with_name("config.json").read_text(encoding="utf-8")
        )
        merged = merge_nsf_weights(aio_checkpoint, vocoder_checkpoint, config, metadata)
        mini["enabled"] = False
        aio_settings["embedded_nsf"] = metadata
    else:
        raise ValueError("Expected a standalone Mini-DDSP or NSF-HiFiGAN generator")
    config.pop("all_in_one_vocoder_backend", None)
    merged["vocoder_merge"].update(aio=str(aio_path), vocoder=str(vocoder_path))

    filenames = [f"dictionary-{language}.txt" for language in config["dictionaries"]]
    if config.get("use_spk_id", False):
        filenames.append("spk_map.json")
    if config.get("use_lang_id", False):
        filenames.append("lang_map.json")
    for name in filenames:
        if not (aio_path.parent / name).is_file():
            raise FileNotFoundError(
                f"Missing AIO inference asset: {aio_path.parent / name}"
            )

    config.update(
        base_config=[],
        work_dir=str(output_dir),
        exp_name=output_dir.name,
        finetune_enabled=False,
        finetune_ckpt_path=None,
    )
    output_dir.mkdir(parents=True)
    output_path = output_dir / "model_ckpt_steps_0.ckpt"
    torch.save(merged, output_path)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for name in filenames:
        shutil.copy2(aio_path.parent / name, output_dir / name)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aio", required=True, type=Path, help="Full AIO training checkpoint"
    )
    parser.add_argument(
        "--vocoder",
        required=True,
        type=Path,
        help="Standalone Mini-DDSP or NSF-HiFiGAN generator checkpoint",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="New experiment directory under ckpt"
    )
    parser.add_argument(
        "--config", type=Path, help="AIO config; defaults to config.yaml beside --aio"
    )
    args = parser.parse_args()
    output = merge_experiment(args.aio, args.vocoder, args.out, args.config)
    print(f"| Merged AIO checkpoint: {output}")
    print("| Acoustic/variance weights unchanged; optimizer state omitted.")
    print(
        "| Use for inference or finetune initialization in a NEW experiment, not in-place resume."
    )


if __name__ == "__main__":
    main()
