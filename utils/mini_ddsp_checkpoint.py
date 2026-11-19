"""Audio and model contracts for independently trained Mini-DDSP weights."""

import torch

from utils.hparams import mini_ddsp_config

AUDIO_CONTRACT_KEYS = (
    "audio_sample_rate",
    "audio_num_mel_bins",
    "hop_size",
    "fft_size",
    "win_size",
    "fmin",
    "fmax",
    "mel_base",
    "pe",
    "f0_min",
    "f0_max",
)


def mini_ddsp_contract(config):
    settings = mini_ddsp_config(config)
    if not settings.get("enabled", False):
        raise ValueError("Mini-DDSP must be enabled")
    contract = {key: config[key] for key in AUDIO_CONTRACT_KEYS}
    contract["model"] = {
        key: int(settings.get(key, default))
        for key, default in (
            ("harmonics", 128),
            ("noise_bands", 64),
            ("hidden_size", 256),
            ("num_layers", 3),
            ("kernel_size", 31),
        )
    }
    return contract


def validate_audio_contract(recorded, config):
    for key in AUDIO_CONTRACT_KEYS:
        if recorded.get(key) != config[key]:
            raise ValueError(f"Vocoder audio contract disagrees on {key}")


def merge_vocoder_weights(aio_checkpoint, vocoder_checkpoint, config):
    """Replace only the vocoder tensors, producing a fresh inference/finetune state."""
    if aio_checkpoint.get("category") != "all_in_one":
        raise ValueError("The destination checkpoint must be all_in_one")
    if vocoder_checkpoint.get("category") != "vocoder":
        raise ValueError("The source checkpoint must be a standalone Mini-DDSP vocoder")
    expected_contract = mini_ddsp_contract(config)
    if vocoder_checkpoint.get("mini_ddsp_contract") != expected_contract:
        raise ValueError("Standalone vocoder audio/model contract does not match AIO")
    source = {
        key.removeprefix("model."): value
        for key, value in vocoder_checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    prefix = "model.vocoder."
    original = aio_checkpoint["state_dict"]
    expected = {
        key.removeprefix(prefix): value
        for key, value in original.items()
        if key.startswith(prefix)
    }
    if not expected:
        raise ValueError("AIO checkpoint must contain a complete built-in Mini-DDSP")
    if source.keys() != expected.keys():
        raise ValueError(
            "Vocoder state keys differ: "
            f"missing={sorted(expected.keys() - source.keys())}, "
            f"unexpected={sorted(source.keys() - expected.keys())}"
        )
    for name, tensor in source.items():
        target = expected[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape:
            raise ValueError(f"Vocoder tensor shape mismatch: {name}")
        if tensor.dtype != target.dtype:
            raise ValueError(f"Vocoder tensor dtype mismatch: {name}")
    merged = original.copy()
    merged.update({prefix + name: tensor for name, tensor in source.items()})
    return {
        "category": "all_in_one",
        "state_dict": merged,
        "global_step": 0,
        "mini_ddsp_contract": expected_contract,
        "vocoder_merge": {
            "aio_step": aio_checkpoint.get("global_step"),
            "vocoder_step": vocoder_checkpoint.get("global_step"),
            "purpose": "inference or initialization of a new finetune experiment",
        },
    }
