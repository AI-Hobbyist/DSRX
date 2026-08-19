"""Export a DSRX training checkpoint as an inference-only checkpoint.

The exported file keeps the exact model state and category while dropping
optimizer, scheduler, callback and trainer state.  Inference automatically
prefers ``model_ckpt_steps_N.infer.ckpt`` beside the original checkpoint.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch


ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _resolve_source(args: argparse.Namespace) -> Path:
    if args.input is not None:
        source = args.input.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source}")
        return source
    experiment = (ROOT / "ckpt" / args.exp).resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment}")
    if args.ckpt is not None:
        source = experiment / f"model_ckpt_steps_{args.ckpt}.ckpt"
        if not source.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {source}")
        return source
    candidates = sorted(
        (
            path
            for path in experiment.iterdir()
            if path.is_file()
            and re.fullmatch(r"model_ckpt_steps_\d+\.ckpt", path.name)
        ),
        key=lambda path: int(re.search(r"\d+", path.name).group()),
    )
    if not candidates:
        raise FileNotFoundError(f"No training checkpoint found in: {experiment}")
    return candidates[-1]


def _resolve_target(source: Path, output: Path | None) -> Path:
    default_name = source.with_suffix(".infer.ckpt").name
    if output is None:
        return source.with_name(default_name)
    output = output.resolve()
    if output.suffix == ".ckpt":
        return output
    return output / default_name


def _verify_state(source: Dict[str, Any], exported: Dict[str, Any]) -> None:
    if source.keys() != exported.keys():
        raise RuntimeError("Exported state_dict keys do not match the source")
    for key in source:
        left = source[key]
        right = exported[key]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if (
                left.shape != right.shape
                or left.dtype != right.dtype
                or not torch.equal(left, right)
            ):
                raise RuntimeError(f"Export verification failed for tensor: {key}")
        elif left != right:
            raise RuntimeError(f"Export verification failed for value: {key}")


def _asset_pairs(source_dir: Path, target_dir: Path):
    if source_dir == target_dir:
        return []
    result = []
    for source in source_dir.iterdir():
        if not source.is_file():
            continue
        if source.suffix in {".yaml", ".json", ".txt"}:
            result.append((source, target_dir / source.name))
    return result


def _preflight_assets(source_dir: Path, target_dir: Path, force: bool) -> None:
    for source, target in _asset_pairs(source_dir, target_dir):
        if (
            target.is_file()
            and not filecmp.cmp(source, target, shallow=False)
            and not force
        ):
            raise FileExistsError(
                f"Asset already exists with different content (use --force): {target}"
            )


def _copy_assets(source_dir: Path, target_dir: Path) -> None:
    for source, target in _asset_pairs(source_dir, target_dir):
        if target.is_file() and filecmp.cmp(source, target, shallow=False):
            continue
        shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop training-only state and export an exact DSRX inference checkpoint."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Source training checkpoint")
    source.add_argument("--exp", type=str, help="Experiment directory name under ckpt/")
    parser.add_argument("--ckpt", type=int, help="Checkpoint step used with --exp")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .ckpt path or directory; defaults beside source as *.infer.ckpt",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing export")
    parser.add_argument(
        "--no-copy-assets",
        action="store_true",
        help="Do not copy yaml/json/txt inference assets when exporting to another directory",
    )
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is not None and args.ckpt is not None:
        raise SystemExit("--ckpt can only be used with --exp")
    source = _resolve_source(args)
    target = _resolve_target(source, args.output)
    manifest_path = args.manifest.resolve() if args.manifest is not None else None
    if source.resolve() == target.resolve():
        raise SystemExit("Refusing to overwrite the source training checkpoint")
    if manifest_path is not None and manifest_path in {source.resolve(), target.resolve()}:
        raise SystemExit("Manifest path must differ from the source and exported checkpoint")
    if target.exists() and not args.force:
        raise SystemExit(f"Output already exists (use --force): {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_assets:
        _preflight_assets(source.parent, target.parent, args.force)
    if manifest_path is not None and manifest_path.exists() and not args.force:
        raise SystemExit(f"Manifest already exists (use --force): {manifest_path}")

    checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise TypeError("Expected a checkpoint dict containing state_dict")
    exported = {"state_dict": checkpoint["state_dict"]}
    if "category" in checkpoint:
        exported["category"] = checkpoint["category"]

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(exported, temporary_path)
        verified = torch.load(temporary_path, map_location="cpu")
        _verify_state(checkpoint["state_dict"], verified["state_dict"])
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if not args.no_copy_assets:
        _copy_assets(source.parent, target.parent)

    manifest = {
        "source": str(source),
        "target": str(target),
        "source_bytes": source.stat().st_size,
        "target_bytes": target.stat().st_size,
        "bytes_saved": source.stat().st_size - target.stat().st_size,
        "size_ratio": target.stat().st_size / source.stat().st_size,
        "state_entries": len(exported["state_dict"]),
        "state_tensor_bytes": _tensor_bytes(exported["state_dict"]),
        "source_top_level_keys": list(checkpoint),
        "exported_top_level_keys": list(exported),
        "source_sha256": _sha256(source),
        "target_sha256": _sha256(target),
        "exact_state_match": True,
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("| inference will automatically prefer this .infer.ckpt file")


if __name__ == "__main__":
    main()
