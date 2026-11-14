import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from preprocessing.all_in_one_binarizer import AllInOneBinarizer
from preprocessing.vocoder_binarizer import VocoderBinarizer
from training.mini_ddsp import MiniDDSPLoss, augment_vocoder_volume
from tests.test_mini_ddsp import config
from modules.vocoders.mini_ddsp import MiniDDSP
from types import SimpleNamespace
from utils.hparams import hparams, set_hparams


class IndependentVocoderTests(unittest.TestCase):
    def setUp(self):
        self.previous = copy.deepcopy(hparams)

    def tearDown(self):
        hparams.clear()
        hparams.update(self.previous)

    def test_enabled_acoustic_pipeline_never_runs_vocoder(self):
        hparams.update(
            {
                "binary_data_dir": "unused",
                "all_in_one": {"enabled": True, "mini_ddsp": {"enabled": True}},
            }
        )
        with patch(
            "preprocessing.all_in_one_binarizer.AcousticBinarizer"
        ) as acoustic, patch(
            "preprocessing.all_in_one_binarizer.VarianceBinarizer"
        ) as variance, patch.object(
            VocoderBinarizer, "process", side_effect=AssertionError("vocoder ran")
        ):
            AllInOneBinarizer().process()
        acoustic.return_value.process.assert_called_once()
        variance.return_value.process.assert_called_once()

    def test_standalone_configuration_and_audio_only_binary_package(self):
        settings = set_hparams(
            config="configs/vocoder/mini_ddsp_opencpop.yaml",
            print_hparams=False,
            global_hparams=False,
        )
        self.assertNotIn("task_cls", settings)
        self.assertNotIn("dictionaries", settings)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "wavs").mkdir()
            for name in ("train", "valid"):
                (root / "wavs" / f"{name}.wav").touch()
            settings["datasets"] = [
                {
                    "raw_data_dir": str(root),
                    "speaker": "speaker",
                    "test_prefixes": ["valid"],
                }
            ]
            settings["binary_data_dir"] = str(root / "binary")
            settings["binarization_args"]["num_workers"] = 0
            hparams.clear()
            hparams.update(settings)

            def extracted(source):
                return {
                    "sample_id": np.bytes_(source["sample_id"]),
                    "augmentation_id": np.bytes_(source["augmentation_id"]),
                    "length": 2,
                    "spk_id": 0,
                    "waveform": np.ones(1024, np.float32),
                }

            with patch.object(VocoderBinarizer, "process_item", side_effect=extracted):
                VocoderBinarizer().process()
            self.assertEqual(
                {path.name for path in (root / "binary" / "vocoder").iterdir()},
                {
                    "audio_contract.json",
                    "train.data",
                    "train.meta",
                    "valid.data",
                    "valid.meta",
                },
            )

    def test_augmented_acoustic_length_does_not_drive_vocoder_supervision(self):
        settings = config()
        settings["win_size"] = 32
        model = MiniDDSP(settings)
        loss = MiniDDSPLoss(settings)
        settings["all_in_one"]["mini_ddsp"].update(warmup_steps=0, joint_ramp_steps=0)
        sample = {
            "size": 1,
            "mel": torch.zeros(1, 2, 8),
            "f0": torch.full((1, 2), 440.0),
            "vocoder": {
                "mel": torch.full((1, 5, 8), -6.0),
                "f0": torch.full((1, 5), 220.0),
                "paired": torch.tensor([False]),
                "frame_lengths": torch.tensor([5]),
                "sample_lengths": torch.tensor([80]),
                "waveform": torch.zeros(1, 80),
                "amplitude": torch.zeros(1, 5, 7),
                "uv": torch.ones(1, 5, dtype=torch.bool),
            },
        }
        output = SimpleNamespace(
            clean_out=torch.randn(1, 2, 8, requires_grad=True), aux_out=None
        )
        losses = loss(model, sample, output, None, 10000, training=False)
        self.assertEqual(losses["ddsp_joint_loss"].item(), 0.0)
        sum(losses.values()).backward()
        self.assertIsNone(output.clean_out.grad)
        self.assertGreater(model.output.bias.grad.abs().sum().item(), 0)

    def test_volume_augmentation_scales_all_supervision_without_mutating_cache(self):
        import math

        mel = torch.zeros(1, 4, 8)
        waveform = torch.full((1, 64), 0.2)
        amplitude = torch.full((1, 4, 7), 0.1)
        with patch("training.mini_ddsp.random.uniform", return_value=math.log(2.0)):
            changed = augment_vocoder_volume(mel, waveform, amplitude, {}, True)
        torch.testing.assert_close(changed[0], mel + math.log(2.0))
        torch.testing.assert_close(changed[1], waveform * 2)
        torch.testing.assert_close(changed[2], amplitude * 2)
        self.assertEqual(mel.sum().item(), 0.0)
        with patch(
            "training.mini_ddsp.random.uniform",
            side_effect=AssertionError("validation augmented"),
        ):
            unchanged = augment_vocoder_volume(mel, waveform, amplitude, {}, False)
            disabled = augment_vocoder_volume(
                mel, waveform, amplitude, {"volume_enabled": False}, True
            )
        self.assertIs(unchanged[0], mel)
        self.assertIs(disabled[1], waveform)

    def test_standalone_uses_configured_pitch_extractor_and_separator(self):
        set_hparams(
            config="configs/vocoder/mini_ddsp_opencpop.yaml", print_hparams=False
        )
        hparams["hnsep"] = "world"
        with patch(
            "preprocessing.vocoder_binarizer.librosa.load",
            return_value=(np.ones(5120, np.float32), 44100),
        ), patch(
            "preprocessing.vocoder_binarizer.get_mel_torch",
            return_value=np.zeros((10, 128), np.float32),
        ), patch(
            "preprocessing.vocoder_binarizer.pitch_extractor", None
        ), patch(
            "preprocessing.vocoder_binarizer.initialize_pe"
        ) as initialize, patch(
            "preprocessing.vocoder_binarizer.DecomposedWaveform"
        ) as separator, patch(
            "preprocessing.vocoder_binarizer.harmonic_amplitudes",
            return_value=np.zeros((10, 128), np.float32),
        ):
            initialize.return_value.get_pitch.return_value = (
                np.full(10, 220, np.float32),
                np.zeros(10, bool),
            )
            separator.return_value.harmonic.return_value = np.ones(5120, np.float32)
            result = VocoderBinarizer().process_item(
                {
                    "path": "audio.wav",
                    "sample_id": "id",
                    "augmentation_id": "raw",
                    "spk_id": 0,
                }
            )
        initialize.assert_called_once()
        self.assertEqual(separator.call_args.kwargs["algorithm"], "world")
        self.assertEqual(
            initialize.return_value.get_pitch.call_args.kwargs["f0_min"],
            hparams["f0_min"],
        )
        self.assertEqual(result["waveform"].shape, (5120,))


if __name__ == "__main__":
    unittest.main()
