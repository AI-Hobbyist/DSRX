"""Independent Mini-DDSP training on the existing audio-only indexed package."""

import json
from pathlib import Path
import shutil

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only

import utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from modules.vocoders.mini_ddsp import MiniDDSP
from training.mini_ddsp import MiniDDSPLoss
from utils.hparams import hparams
from utils.mini_ddsp_checkpoint import mini_ddsp_contract, validate_audio_contract


class VocoderDataset(BaseDataset):
    def __init__(self, prefix, preload=False, data_dir=None):
        directory = Path(data_dir or hparams["binary_data_dir"]) / "vocoder"
        contract = json.loads(
            (directory / "audio_contract.json").read_text(encoding="utf-8")
        )
        validate_audio_contract(contract, hparams)
        super().__init__(prefix, preload=preload, data_dir=directory)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if item["sample_rate"] != hparams["audio_sample_rate"]:
            raise ValueError(
                "Vocoder sample rate disagrees with training configuration"
            )
        harmonics = mini_ddsp_contract(hparams)["model"]["harmonics"]
        if item["amplitude"].shape[-1] != harmonics:
            raise ValueError(
                "Vocoder amplitude labels disagree on harmonics; rebuild vocoder data"
            )
        return item

    def collater(self, samples):
        batch = super().collater(samples)
        if not samples:
            return batch
        paired = {
            key: utils.collate_nd([item[key] for item in samples], padding)
            for key, padding in (
                ("mel", 0.0),
                ("f0", 0.0),
                ("waveform", 0.0),
                ("amplitude", 0.0),
                ("uv", True),
            )
        }
        paired["sample_lengths"] = torch.tensor(
            [item["sample_length"] for item in samples]
        )
        paired["frame_lengths"] = torch.tensor([item["length"] for item in samples])
        batch["vocoder"] = paired
        return batch


class MiniDDSPTask(BaseTask):
    requires_phoneme_dictionary = False

    def __init__(self):
        if hparams.get("lora", {}).get("enabled", False):
            raise ValueError("Independent Mini-DDSP training does not use LoRA")
        super().__init__()
        self.dataset_cls = VocoderDataset
        self._finish_init()

    def _build_model(self):
        mini_ddsp_contract(hparams)
        return MiniDDSP(hparams)

    def build_losses_and_metrics(self):
        self.ddsp_loss = MiniDDSPLoss(hparams)
        for name in ("ddsp_spectral_loss", "ddsp_amplitude_loss"):
            self.register_validation_loss(name)

    def run_model(self, sample, infer=False):
        if infer:
            paired = sample["vocoder"]
            return self.model(
                paired["mel"], paired["f0"], lengths=paired["frame_lengths"]
            )
        return self.ddsp_loss(
            self.model,
            sample,
            output=None,
            acoustic=None,
            step=self.global_step,
            training=self.training,
            include_joint=False,
        )

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample)
        if batch_idx < hparams["num_valid_plots"]:
            with torch.no_grad():
                waveform = self.run_model(sample, infer=True)
            for index in range(sample["size"]):
                count = int(sample["vocoder"]["sample_lengths"][index])
                tag = f"vocoder_{int(sample['indices'][index])}"
                for name, audio in (
                    ("reference", sample["vocoder"]["waveform"]),
                    ("prediction", waveform),
                ):
                    self.logger.all_rank_experiment.add_audio(
                        f"{tag}/{name}",
                        audio[index, :count],
                        sample_rate=hparams["audio_sample_rate"],
                        global_step=self.global_step,
                    )
        return losses, sample["size"]

    @rank_zero_only
    def copy_training_payload(self, work_dir):
        source = Path(hparams["binary_data_dir"]) / "vocoder" / "audio_contract.json"
        shutil.copy(source, work_dir / "audio_contract.json")

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["category"] = "vocoder"
        checkpoint["mini_ddsp_contract"] = mini_ddsp_contract(hparams)

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("category") != "vocoder":
            raise ValueError("Independent training requires a vocoder checkpoint")
        if checkpoint.get("mini_ddsp_contract") != mini_ddsp_contract(hparams):
            raise ValueError(
                "Vocoder checkpoint audio/model contract does not match configuration"
            )
        super().on_load_checkpoint(checkpoint)
