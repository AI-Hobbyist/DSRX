"""Standalone Mini-DDSP adapter for the existing vocoder registry."""

import torch

from modules.vocoders.all_in_one import AllInOneVocoder
from modules.vocoders.mini_ddsp import MiniDDSP
from modules.vocoders.registry import register_vocoder
from utils.hparams import hparams
from utils.mini_ddsp_checkpoint import validate_audio_contract


@register_vocoder
class MiniDDSPVocoder(AllInOneVocoder):
    def __init__(self):
        checkpoint = torch.load(
            hparams["vocoder_ckpt"], map_location="cpu", weights_only=False
        )
        contract = checkpoint.get("mini_ddsp_contract")
        if checkpoint.get("category") != "vocoder" or not isinstance(
            contract, dict
        ):
            raise ValueError(
                "vocoder_ckpt must be a standalone Mini-DDSP checkpoint"
            )
        validate_audio_contract(contract, hparams)
        vocoder_config = {
            **hparams,
            "all_in_one": {
                "enabled": True,
                "mini_ddsp": {**contract["model"], "enabled": True},
            },
        }
        vocoder = MiniDDSP(vocoder_config)
        vocoder.load_state_dict(
            {
                key.removeprefix("model."): tensor
                for key, tensor in checkpoint["state_dict"].items()
                if key.startswith("model.")
            },
            strict=True,
        )
        vocoder.requires_grad_(False).eval()
        super().__init__(vocoder, internal=True)
