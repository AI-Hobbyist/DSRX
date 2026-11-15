"""Bounded real-data joint training, gradient isolation and resume validation."""

import argparse
import copy
import json
import pathlib

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.nn import functional as F
import yaml

from training.acoustic_task import AcousticTask
from training.all_in_one_task import AllInOneTask
from training.mini_ddsp import PairedAcousticDataset, joint_factor
from utils.hparams import hparams, set_hparams


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


class TrainingEvidence(pl.Callback):
    def __init__(self, checkpoints):
        self.steps = []
        self.checkpoints = checkpoints
        self.resume_expected = None

    def on_train_start(self, trainer, task):
        if self.resume_expected is not None:
            actual = trainer.optimizers[0].state_dict()["state"]
            for key, values in self.resume_expected["state"].items():
                for name, expected in values.items():
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(
                            actual[key][name].cpu(), expected.cpu(), rtol=0, atol=0
                        )

    def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx):
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite real-data joint training loss")
        self.steps.append(
            {
                "step": trainer.global_step,
                "loss": float(loss.detach().cpu()),
                "joint_factor": joint_factor(
                    trainer.global_step - 1, task.mini_ddsp_settings
                ),
            }
        )
        if trainer.global_step in self.checkpoints:
            trainer.save_checkpoint(self.checkpoints[trainer.global_step])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/mini_ddsp_m2/singing_on/config.yaml"),
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("artifacts/mini_ddsp_m3")
    )
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    initial_config = yaml.safe_load(
        pathlib.Path("ckpt/dslight/config.yaml").read_text(encoding="utf-8")
    )
    for name in ("energy", "breathiness", "voicing", "tension"):
        config[f"use_{name}_embed"] = initial_config[f"use_{name}_embed"]
        config[f"predict_{name}"] = initial_config[f"predict_{name}"]
    config.update(
        max_sample_frames=4096,
        max_batch_size=2,
        max_batch_frames=8192,
        max_val_batch_size=1,
        ds_workers=0,
        accumulate_grad_batches=1,
        val_with_vocoder=False,
        val_with_variance={"enable": False},
        num_valid_plots=0,
        finetune_enabled=False,
        freezing_enabled=False,
        infer=False,
        optimizer_args={
            "optimizer_cls": "torch.optim.AdamW",
            "lr": 1e-4,
            "weight_decay": 0.0,
        },
    )
    config["all_in_one"]["mini_ddsp"].update(
        hidden_size=256,
        num_layers=3,
        kernel_size=31,
        noise_bands=64,
        segment_frames=32,
        warmup_steps=2,
        joint_ramp_steps=4,
    )
    config["work_dir"] = str(args.output)
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    set_hparams(config=str(config_path), print_hparams=False)
    hparams["work_dir"] = str(args.output)
    task = AllInOneTask()
    initial_path = pathlib.Path("ckpt/dslight/model_ckpt_steps_100000.ckpt")
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    task.load_finetune_ckpt(initial["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task.to(device)
    task.eval()
    dataset = PairedAcousticDataset(
        "train", data_dir=pathlib.Path(config["binary_data_dir"]) / "acoustic"
    )
    sample = to_device(dataset.collater([dataset[0]]), device)
    _, output = AcousticTask.run_model(
        task, sample, model=task.model.acoustic, return_output=True
    )
    waveform_losses = task.ddsp_loss(
        task.model.vocoder, sample, output, task.model.acoustic, 6, training=False
    )
    waveform_losses["ddsp_joint_loss"].backward()
    gradients = {}
    for label, prefix in (
        ("vocoder", "vocoder."),
        ("acoustic_encoder", "acoustic.fs2.encoder."),
        ("acoustic_token_embedding", "acoustic.fs2.txt_embed."),
        ("main_predictor", "acoustic.diffusion.velocity_fn."),
    ):
        values = [
            parameter.grad
            for name, parameter in task.model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        assert values and all(torch.isfinite(value).all() for value in values), label
        norm = sum(float(value.abs().sum()) for value in values)
        assert norm > 0, label
        gradients[label] = {"nonzero_finite_l1": norm, "tensors": len(values)}
    # Only padded tails change. The exact same valid region must give the same
    # waveform and amplitude losses, even with nonzero garbage in the padding.
    padded = copy.deepcopy(sample)
    for name in ("mel", "f0"):
        padding = (0, 0, 0, 7) if name == "mel" else (0, 7)
        padded[name] = F.pad(padded[name], padding, value=123)
    padded["vocoder"]["waveform"] = F.pad(
        padded["vocoder"]["waveform"], (0, 7 * config["hop_size"]), value=123
    )
    padded["vocoder"]["amplitude"] = F.pad(
        padded["vocoder"]["amplitude"], (0, 0, 0, 7), value=123
    )
    padded["vocoder"]["uv"] = F.pad(padded["vocoder"]["uv"], (0, 7), value=False)
    with torch.no_grad():
        changed = task.ddsp_loss(
            task.model.vocoder, padded, output, task.model.acoustic, 6, training=False
        )
        for key in waveform_losses:
            torch.testing.assert_close(
                waveform_losses[key], changed[key], rtol=0, atol=0
            )
    task.zero_grad(set_to_none=True)
    del output, waveform_losses, changed, sample, padded, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    first_steps = args.steps // 2
    resume_path = args.output / "resume.ckpt"
    final_path = args.output / f"model_ckpt_steps_{args.steps}.ckpt"
    evidence = TrainingEvidence({first_steps: resume_path, args.steps: final_path})

    def trainer(max_steps):
        return pl.Trainer(
            accelerator="gpu" if device.type == "cuda" else "cpu",
            devices=1,
            precision="32-true",
            max_steps=max_steps,
            max_epochs=-1,
            logger=CSVLogger(str(args.output), name="training"),
            callbacks=[evidence],
            enable_checkpointing=False,
            enable_model_summary=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=config["accumulate_grad_batches"],
            gradient_clip_val=1.0,
        )

    first = trainer(first_steps)
    first.fit(task)
    saved = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert saved["global_step"] == first_steps and saved["optimizer_states"]
    evidence.resume_expected = saved["optimizer_states"][0]
    restored = AllInOneTask()
    second = trainer(args.steps)
    second.fit(restored, ckpt_path=str(resume_path))
    assert second.global_step == args.steps
    resumed = [entry for entry in evidence.steps if entry["step"] == first_steps + 1]
    assert resumed[0]["joint_factor"] == joint_factor(
        first_steps, config["all_in_one"]["mini_ddsp"]
    )
    validation = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        logger=CSVLogger(str(args.output), name="validation"),
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_val_batches=1,
    )
    validation.validate(restored)
    validation_loss = float(validation.callback_metrics["val_loss"].cpu())
    assert torch.isfinite(torch.tensor(validation_loss))
    report = {
        "seed": 42,
        "initial_checkpoint": str(initial_path),
        "steps": args.steps,
        "resume_step": first_steps,
        "gradients_waveform_only": gradients,
        "padding_exact": True,
        "optimizer_restored": True,
        "training_stream_count": 2,
        "validation_loss": validation_loss,
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "history": evidence.steps,
        "checkpoint": str(final_path),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
