import copy
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from inference.ds_acoustic import DiffSingerAcousticInfer
from modules.vocoders.all_in_one import (
    AllInOneVocoder,
    build_all_in_one_vocoder,
    validate_nsf_contract,
)
from modules.vocoders.mini_ddsp import MiniDDSP
from tests.test_mini_ddsp import config
from utils.hparams import hparams


class MiniDDSPInferenceTests(unittest.TestCase):
    def setUp(self):
        self.previous = copy.deepcopy(hparams)
        hparams.clear()
        hparams.update(config())

    def tearDown(self):
        hparams.clear()
        hparams.update(self.previous)

    def test_internal_default_needs_no_external_asset_and_preserves_batch(self):
        model = SimpleNamespace(vocoder=MiniDDSP(hparams))
        hparams["vocoder_ckpt"] = "a-path-that-does-not-exist"
        with patch(
            "modules.vocoders.nsf_hifigan.NsfHifiGAN",
            side_effect=AssertionError("external load"),
        ):
            backend = build_all_in_one_vocoder(model)
            mel = torch.randn(2, 5, 8)
            f0 = torch.full((2, 5), 220.0)
            actual = DiffSingerAcousticInfer.run_vocoder(
                SimpleNamespace(vocoder=backend), mel, f0=f0
            )
        self.assertEqual(actual.shape, (2, 80))
        torch.testing.assert_close(actual, model.vocoder(mel, f0), rtol=0, atol=0)

    def test_external_switch_receives_identical_features_without_batch_flatten(self):
        class External:
            def __init__(self):
                self.inputs = []

            def spec2wav_torch(self, mel, f0):
                self.inputs.append((mel.clone(), f0.clone()))
                return torch.ones(mel.shape[1] * 16)

        external = External()
        backend = AllInOneVocoder(external, internal=False)
        mel = torch.randn(2, 5, 8)
        f0 = torch.tensor([[220.0] * 5, [330.0] * 5])
        self.assertEqual(backend.spec2wav_torch(mel, f0).shape, (2, 80))
        for index, (received_mel, received_f0) in enumerate(external.inputs):
            torch.testing.assert_close(
                received_mel, mel[index : index + 1], rtol=0, atol=0
            )
            torch.testing.assert_close(
                received_f0, f0[index : index + 1], rtol=0, atol=0
            )

    def test_nsf_contract_rejects_every_mismatch(self):
        source = {
            "audio_sample_rate": 44100,
            "hop_size": 512,
            "audio_num_mel_bins": 128,
            "fft_size": 2048,
            "win_size": 2048,
            "fmin": 40,
            "fmax": 16000,
            "mel_base": "e",
        }
        metadata = {
            "sampling_rate": 44100,
            "hop_size": 512,
            "num_mels": 128,
            "n_fft": 2048,
            "win_size": 2048,
            "fmin": 40,
            "fmax": 16000,
        }
        validate_nsf_contract(source, metadata)
        for key in source:
            with self.subTest(key=key):
                changed = {**source, key: 10 if key == "mel_base" else source[key] + 1}
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    validate_nsf_contract(changed, metadata)

    def test_shared_offsets_silence_and_crossfade(self):
        hparams["audio_sample_rate"] = 100
        captured = []

        class Infer:
            def preprocess_input(self, param, idx):
                return {"f0": torch.zeros(1, 10), "value": param["value"]}

            def forward_model(self, batch):
                return torch.full((1, 10, 1), batch["value"])

            def run_vocoder(self, mel, **kwargs):
                return mel[:, :, 0]

        params = [
            {"offset": 0.10, "value": 1.0},
            {"offset": 0.25, "value": 2.0},
            {"offset": 0.30, "value": 3.0},
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "inference.ds_acoustic.save_wav",
            side_effect=lambda waveform, *args: captured.append(waveform),
        ):
            DiffSingerAcousticInfer.run_inference(
                Infer(), params, out_dir=pathlib.Path(directory), title="segments"
            )
        waveform = captured[0]
        self.assertEqual(waveform.shape, (40,))
        for start, stop, value in (
            (0, 10, 0),
            (10, 20, 1),
            (20, 25, 0),
            (25, 30, 2),
            (35, 40, 3),
        ):
            np.testing.assert_array_equal(waveform[start:stop], value)
        self.assertTrue(((waveform[30:35] >= 2) & (waveform[30:35] <= 3)).all())


if __name__ == "__main__":
    unittest.main()
