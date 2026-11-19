"""Package independent Mini-DDSP weights into a new AIO inference experiment."""

import argparse
from pathlib import Path
import shutil
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hparams import set_hparams
from utils.mini_ddsp_checkpoint import merge_vocoder_weights


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
    merged = merge_vocoder_weights(aio_checkpoint, vocoder_checkpoint, config)
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
        "--vocoder", required=True, type=Path, help="Independent Mini-DDSP checkpoint"
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
