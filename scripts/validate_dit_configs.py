from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPLACE_CONFIG_MAPPINGS = {
    "aux_datasets", "backbone_args", "optimizer_args", "lr_scheduler_args"
}
DIT_ARGS = {
    "num_layers",
    "num_channels",
    "num_heads",
    "mlp_ratio",
    "time_embed_dim",
    "patch_size",
    "rope_base",
    "layer_norm_eps",
    "attention_dropout",
    "mlp_dropout",
    "use_gradient_checkpointing",
}


def override_config(old_config: dict, new_config: dict) -> None:
    for key, value in new_config.items():
        old_value = old_config.get(key)
        if (
            key not in REPLACE_CONFIG_MAPPINGS
            and isinstance(value, dict)
            and isinstance(old_value, dict)
        ):
            override_config(old_value, value)
        else:
            old_config[key] = value


def load_config(config_path: Path, loaded: set[Path] | None = None) -> dict:
    loaded = set() if loaded is None else loaded
    config_path = config_path.resolve()
    if config_path in loaded:
        return {}
    loaded.add(config_path)

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    flattened = {}
    base_configs = config.get("base_config", [])
    if not isinstance(base_configs, list):
        base_configs = [base_configs]
    for base_config in base_configs:
        base_path = (
            (config_path.parent / base_config).resolve()
            if str(base_config).startswith(".")
            else (ROOT / base_config).resolve()
        )
        override_config(flattened, load_config(base_path, loaded))
    override_config(flattened, config)
    return flattened


def validate_dit_args(name: str, args: object, expected_channels: int) -> None:
    if not isinstance(args, dict):
        raise AssertionError(f"{name}: backbone_args must be a mapping")
    unknown = set(args) - DIT_ARGS
    missing = DIT_ARGS - set(args)
    assert not unknown, f"{name}: unknown DiT arguments: {sorted(unknown)}"
    assert not missing, f"{name}: missing DiT arguments: {sorted(missing)}"
    assert args["num_channels"] == expected_channels
    assert args["num_channels"] % args["num_heads"] == 0
    assert args["num_channels"] // args["num_heads"] == 64
    assert args["patch_size"] == 1


def validate_common(
    config: dict, expected_batch_frames: int = 768,
    expected_batch_size: int = 1
) -> None:
    contract = config["dit_contract"]
    assert contract == {
        "version": 1,
        "valid_mask_source": "mel2ph",
        "valid_mask_true_is_valid": True,
        "padded_batches_require_mask": True,
    }
    assert config["max_batch_frames"] == expected_batch_frames
    assert config["max_batch_size"] == expected_batch_size
    assert config["max_sample_frames"] == 768
    assert config["accumulate_grad_batches"] == 8
    assert config["inference_max_frames"] == 2048
    assert config["inference_dit_warmup_frames"] == [128, 512, 768, 1024]
    assert config["lora"]["enabled"] is False
    assert config["lora"]["base_ckpt"] is None
    assert all(".backbone." not in target for target in config["lora"]["target_modules"])


def validate_acoustic_optimization(config: dict) -> None:
    optimizer = config["optimizer_args"]
    assert optimizer == {
        "optimizer_cls": "torch.optim.AdamW",
        "lr": 0.0003,
        "weight_decay": 0.0,
    }
    scheduler = config["lr_scheduler_args"]
    assert set(scheduler) == {"scheduler_cls", "schedulers", "milestones"}
    assert scheduler["scheduler_cls"] == "torch.optim.lr_scheduler.SequentialLR"
    assert scheduler["milestones"] == [1000]
    warmup, decay = scheduler["schedulers"]
    assert warmup == {
        "cls": "torch.optim.lr_scheduler.LinearLR",
        "start_factor": 0.01,
        "end_factor": 1.0,
        "total_iters": 1000,
    }
    assert decay == {
        "cls": "torch.optim.lr_scheduler.StepLR",
        "step_size": 5000,
        "gamma": 0.8,
    }


def main() -> None:
    acoustic = load_config(ROOT / "configs/templates/config_acoustic_dit.yaml")
    variance = load_config(ROOT / "configs/templates/config_variance_dit.yaml")
    acoustic_4090 = load_config(
        ROOT / "configs/templates/config_acoustic_4090_10h_dit.yaml"
    )
    variance_4090 = load_config(
        ROOT / "configs/templates/config_variance_4090_10h_dit.yaml"
    )
    all_in_one = load_config(ROOT / "configs/templates/all_in_one_dit.yaml")
    legacy_acoustic = load_config(ROOT / "configs/original/acoustic.yaml")
    legacy_variance = load_config(ROOT / "configs/original/variance.yaml")
    wavenet_all_in_one = load_config(
        ROOT / "configs/templates/all_in_one_wavenet_adamw.yaml"
    )
    lynxnet2_all_in_one = load_config(
        ROOT / "configs/templates/all_in_one_lynxnet2_muon.yaml"
    )

    assert legacy_acoustic["backbone_type"] == "lynxnet2"
    assert legacy_variance["pitch_prediction_args"]["backbone_type"] == "lynxnet2"
    assert legacy_acoustic["all_in_one"]["enabled"] is False
    assert legacy_variance["all_in_one"]["enabled"] is False

    for config in (wavenet_all_in_one, lynxnet2_all_in_one):
        assert config["all_in_one"]["enabled"] is True
        assert config["max_batch_frames"] == 1536
        assert config["max_batch_size"] == 2
        assert config["task_cls"] == "training.all_in_one_task.AllInOneTask"
        assert config["binarizer_cls"] == "preprocessing.all_in_one_binarizer.AllInOneBinarizer"
        assert config["val_with_variance"]["enable"] is False
        assert all(
            config[f"predict_{name}"] is True
            for name in ("dur", "pitch", "energy", "breathiness", "voicing", "tension")
        )

    assert wavenet_all_in_one["backbone_type"] == "wavenet"
    assert wavenet_all_in_one["pitch_prediction_args"]["backbone_type"] == "wavenet"
    assert wavenet_all_in_one["variances_prediction_args"]["backbone_type"] == "wavenet"
    assert wavenet_all_in_one["optimizer_args"]["optimizer_cls"] == "torch.optim.AdamW"
    assert set(wavenet_all_in_one["optimizer_args"]) == {
        "optimizer_cls", "lr", "betas", "weight_decay"
    }

    assert lynxnet2_all_in_one["backbone_type"] == "lynxnet2"
    assert lynxnet2_all_in_one["pitch_prediction_args"]["backbone_type"] == "lynxnet2"
    assert lynxnet2_all_in_one["variances_prediction_args"]["backbone_type"] == "lynxnet2"
    assert lynxnet2_all_in_one["optimizer_args"]["optimizer_cls"] == (
        "modules.optimizer.muon.Muon_AdamW"
    )

    validate_common(acoustic)
    assert acoustic["backbone_type"] == "dit"
    validate_dit_args("acoustic", acoustic["backbone_args"], 384)
    validate_acoustic_optimization(acoustic)

    validate_common(variance)
    pitch = variance["pitch_prediction_args"]
    variances = variance["variances_prediction_args"]
    assert pitch["backbone_type"] == "dit"
    assert variances["backbone_type"] == "dit"
    validate_dit_args("pitch", pitch["backbone_args"], 256)
    validate_dit_args("variances", variances["backbone_args"], 256)

    validate_common(all_in_one, expected_batch_frames=1536, expected_batch_size=2)
    validate_acoustic_optimization(all_in_one)
    assert all_in_one["all_in_one"]["enabled"] is True
    assert all_in_one["val_with_variance"]["enable"] is False
    assert all_in_one["task_cls"] == "training.all_in_one_task.AllInOneTask"
    assert all_in_one["binarizer_cls"] == "preprocessing.all_in_one_binarizer.AllInOneBinarizer"
    assert all(
        all_in_one[f"predict_{name}"] is True
        for name in ("dur", "pitch", "energy", "breathiness", "voicing", "tension")
    )
    assert all_in_one["backbone_type"] == "dit"
    validate_dit_args("all_in_one.acoustic", all_in_one["backbone_args"], 384)
    validate_dit_args(
        "all_in_one.pitch", all_in_one["pitch_prediction_args"]["backbone_args"], 256
    )
    validate_dit_args(
        "all_in_one.variances",
        all_in_one["variances_prediction_args"]["backbone_args"],
        256
    )

    assert acoustic_4090["backbone_args"] == acoustic["backbone_args"]
    validate_acoustic_optimization(acoustic_4090)
    assert acoustic_4090["max_batch_size"] == 8
    assert acoustic_4090["max_batch_frames"] == 6144
    assert acoustic_4090["max_sample_frames"] == 768
    assert acoustic_4090["accumulate_grad_batches"] == 1
    assert acoustic_4090["val_check_interval"] == 500
    assert acoustic_4090["pl_trainer_precision"] == "bf16-mixed"

    assert variance_4090["pitch_prediction_args"] == variance["pitch_prediction_args"]
    assert variance_4090["variances_prediction_args"] == variance["variances_prediction_args"]
    assert variance_4090["max_batch_size"] == 16
    assert variance_4090["max_batch_frames"] == 12288
    assert variance_4090["max_sample_frames"] == 768
    assert variance_4090["accumulate_grad_batches"] == 1
    assert variance_4090["val_check_interval"] == 500
    assert variance_4090["pl_trainer_precision"] == "bf16-mixed"

    baseline = load_config(ROOT / "configs/dit/p0_baseline.yaml")
    assert baseline["baseline"]["commit"] == "caa651f0a68d59c572915dc3dae2994d288a5f09"
    assert set(baseline["contracts"]["configuration"]["allowed_backbone_args"]) == DIT_ARGS
    assert baseline["contracts"]["configuration"]["unknown_backbone_args"] == "reject"
    assert len(baseline["integration_points"]) == 11
    assert len(baseline["minimal_matrix"]) == 5
    for case in baseline["minimal_matrix"]:
        assert (ROOT / case["config"]).is_file()
        assert (ROOT / case["sample"]).is_file()
    assert baseline["implementation_status"]["p1_backend"] == "complete"
    assert baseline["implementation_status"]["p2_native_inference"] == "complete"
    assert baseline["implementation_status"]["p3_onnx_float"] == "complete"
    assert baseline["implementation_status"]["training_supported"] is True
    assert baseline["implementation_status"]["full_application_inference_supported"] is True
    assert baseline["implementation_status"]["onnx_export_supported"] is True
    prerequisites = baseline["future_phase_prerequisites"]
    assert set(prerequisites) == {"p4", "p5", "p6"}
    print("DiT P0 configuration validation passed.")


if __name__ == "__main__":
    main()