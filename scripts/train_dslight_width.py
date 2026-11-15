"""Run bounded, disjoint-data width adaptation using the supplied DSLight teacher."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_dslight_sampling import acoustic_inputs, load_acoustic
from modules.backbones.dslight import DSLightBackbone
from preprocessing.acoustic_binarizer import AcousticBinarizer
from utils.hparams import hparams


def initialize_narrow(teacher, student):
    """Keep one shared channel subset, including both halves of the gate."""
    old_width = teacher.input_projection.out_channels
    width = student.input_projection.out_channels
    gate_width = student.blocks[0].expand.out_channels // 2
    if width > old_width or (width == old_width and gate_width >= old_width):
        raise ValueError("Reduce the residual or gate width relative to the teacher")
    importance = teacher.output_projection.weight.detach().abs().sum(dim=(0, 2))
    keep = importance.topk(width).indices.sort().values
    gates = torch.cat((keep, keep + old_width))
    source = teacher.state_dict()
    result = {}
    for key, expected in student.state_dict().items():
        value = source[key]
        if key == "output_projection.weight":
            value = value.index_select(1, keep)
        elif key == "output_projection.bias":
            pass
        elif ".expand." in key:
            if width == old_width:
                block = teacher.blocks[int(key.split(".")[1])]
                scores = block.project.weight.detach().abs().sum(dim=(0, 2))
                gate_keep = (
                    (scores[:old_width] + scores[old_width:])
                    .topk(gate_width)
                    .indices.sort()
                    .values
                )
                gates = torch.cat((gate_keep, gate_keep + old_width))
            value = value.index_select(0, gates)
            if value.ndim == 3:
                value = value.index_select(1, keep)
        elif ".project.weight" in key:
            value = value.index_select(0, keep).index_select(1, gates)
        else:
            value = value.index_select(0, keep)
            if value.ndim == 3 and value.shape[1] == old_width:
                value = value.index_select(1, keep)
        if value.shape != expected.shape:
            raise ValueError(f"Incorrect channel mapping for {key}")
        result[key] = value
    student.load_state_dict(result, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--gate-width", type=int)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--train-items", type=int, default=8)
    parser.add_argument(
        "--state-source",
        choices=["interpolation", "teacher-rollout"],
        default="interpolation",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if min(args.width, args.updates, args.train_items) < 1:
        parser.error("Width, updates and training items must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    teacher = load_acoustic(args.config, args.checkpoint, args.device)
    options = dict(hparams["backbone_args"], num_channels=args.width)
    if args.gate_width is not None:
        options["expansion_channels"] = args.gate_width
    student = DSLightBackbone(hparams["audio_num_mel_bins"], 1, **options).to(
        args.device
    )
    initialize_narrow(teacher.diffusion.velocity_fn, student)
    args.output.mkdir(parents=True, exist_ok=True)
    binarizer = AcousticBinarizer()
    binarizer.device = torch.device(args.device)
    samples = []
    names = []
    for dataset_index, dataset in enumerate(hparams["datasets"]):
        metadata = binarizer.load_meta_data(
            Path(dataset["raw_data_dir"]),
            dataset_index,
            dataset["speaker"],
            dataset["language"],
        )
        held_out = tuple(str(x) for x in dataset.get("test_prefixes", []))
        for name, entry in sorted(metadata.items()):
            if held_out and name.split(":", 1)[1].startswith(held_out):
                continue
            item = binarizer.process_item(name, entry, hparams["binarization_args"])
            if item is None:
                raise RuntimeError(f"Training feature extraction failed: {name}")
            inputs = acoustic_inputs(item, args.device)
            with torch.inference_mode():
                condition = teacher.fs2(**inputs).transpose(1, 2).clone()
                target = (
                    teacher.diffusion.norm_spec(
                        torch.as_tensor(item["mel"], device=args.device)[None]
                    )
                    .transpose(-2, -1)[:, None]
                    .clone()
                )
            # Autograd needs ordinary tensors, not inference tensors saved by conv.
            if args.state_source == "teacher-rollout":
                with torch.no_grad():
                    aux = teacher.aux_decoder(condition.transpose(1, 2), infer=True)
                    normalized = teacher.diffusion.norm_spec(aux).transpose(-2, -1)[
                        :, None
                    ]
                    state = 0.4 * normalized + 0.6 * torch.randn_like(normalized)
                    for step_index in range(20):
                        t = torch.tensor([0.4 + 0.03 * step_index], device=args.device)
                        velocity = teacher.diffusion.velocity_fn(
                            state, t * teacher.diffusion.time_scale_factor, condition
                        )
                        samples.append(
                            (condition.clone(), state.clone(), t, velocity.clone())
                        )
                        state = state + 0.03 * velocity
            else:
                samples.append((condition.clone(), target.clone()))
            names.append(name)
            print(f"Prepared training item {name}", flush=True)
            if len(names) >= args.train_items:
                break
        if len(names) >= args.train_items:
            break
    if not samples:
        raise RuntimeError("No disjoint training items available")
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=0.0)
    losses = []
    for update in range(args.updates):
        if args.state_source == "teacher-rollout":
            sample_index = int(torch.randint(len(samples), (1,)).item())
            condition, state, t, velocity = samples[sample_index]
        else:
            condition, target = samples[update % len(samples)]
            t = 0.4 + 0.6 * torch.rand(1, device=args.device)
            noise = torch.randn_like(target)
            state = noise + t[:, None, None, None] * (target - noise)
            with torch.no_grad():
                velocity = teacher.diffusion.velocity_fn(
                    state, t * teacher.diffusion.time_scale_factor, condition
                )
        prediction = student(state, t * teacher.diffusion.time_scale_factor, condition)
        loss = (prediction - velocity).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite width adaptation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if (update + 1) % 50 == 0:
            print(
                f"Update {update + 1}/{args.updates}: loss={loss.item():.6f}",
                flush=True,
            )
    candidate = copy.deepcopy(teacher)
    candidate.diffusion.velocity_fn = student.eval()
    hparams.update(sampling_algorithm="euler", sampling_steps=20, T_start_infer=0.4)
    comparisons = []
    with torch.inference_mode():
        for path in sorted(args.validation_features.glob("sample_*.pt")):
            sample = torch.load(path, map_location=args.device, weights_only=True)
            if sample["name"] in names:
                raise RuntimeError("Training/validation leakage")
            outputs = []
            for model in (teacher, candidate):
                torch.manual_seed(42)
                result = model(**sample["inputs"], infer=True).diff_out
                if not torch.isfinite(result).all():
                    raise RuntimeError("Nonfinite validation mel")
                outputs.append(result)
            comparisons.append(
                {
                    "name": sample["name"],
                    "teacher_target_mel_mae": (outputs[0] - sample["target"])
                    .abs()
                    .mean()
                    .item(),
                    "student_target_mel_mae": (outputs[1] - sample["target"])
                    .abs()
                    .mean()
                    .item(),
                    "student_teacher_mel_mae": (outputs[1] - outputs[0])
                    .abs()
                    .mean()
                    .item(),
                }
            )
    if not comparisons:
        raise RuntimeError("No held-out validation features")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["backbone_args"] = options
    config["all_in_one"] = {"enabled": False}
    for language in hparams["dictionaries"]:
        shutil.copyfile(
            args.config.parent / f"dictionary-{language}.txt",
            args.output / f"dictionary-{language}.txt",
        )
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint_path = args.output / f"model_ckpt_steps_{args.updates}.infer.ckpt"
    torch.save(
        {
            "category": "acoustic",
            "state_dict": {
                f"model.{key}": value.cpu()
                for key, value in candidate.state_dict().items()
            },
        },
        checkpoint_path,
    )
    # Verify the generated configuration and checkpoint actually load strictly.
    load_acoustic(config_path, checkpoint_path, args.device)
    report = {
        "teacher_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "width": args.width,
        "updates": args.updates,
        "training_items": names,
        "gate_width": args.gate_width,
        "state_source": args.state_source,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "validation": comparisons,
        "scope": "Bounded teacher-velocity adaptation; no listening equivalence claim",
    }
    (args.output / "adaptation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
