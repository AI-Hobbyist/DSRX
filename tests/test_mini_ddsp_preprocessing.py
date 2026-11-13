import copy
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from preprocessing.mini_ddsp_cache import pairing_id
from preprocessing.vocoder_binarizer import harmonic_amplitudes
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams


class MiniDDSPPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.previous = copy.deepcopy(hparams)
        self.temporary = tempfile.TemporaryDirectory()
        directory = pathlib.Path(self.temporary.name)
        (directory / "model.pt").write_bytes(b"test separator identity")
        (directory / "config.yaml").write_text("sr: 44100\n")
        hparams.update(
            {
                "all_in_one": {
                    "enabled": True,
                    "mini_ddsp": {"enabled": True, "harmonics": 4},
                },
                "_mini_ddsp_cache_dir": str(directory),
                "hnsep_ckpt": str(directory / "model.pt"),
                "hnsep_sample_rate": 44100,
                "audio_sample_rate": 24000,
                "hop_size": 120,
                "f0_min": 65,
            }
        )

    def tearDown(self):
        hparams.clear()
        hparams.update(self.previous)
        self.temporary.cleanup()

    def test_vr_cache_resamples_and_reuses_full_harmonic(self):
        class Separator:
            is_mono = True

            def __init__(self):
                self.calls = []

            def predict_from_audio(self, x):
                self.calls.append(x.shape[-1])
                return x * 0.75

        model = Separator()
        for rate in (24000, 44100, 48000):
            source = np.sin(np.arange(rate // 10) * 220 * 2 * np.pi / rate).astype(
                np.float32
            )
            with patch("utils.decomposed_waveform.SEP_MODEL", model):
                first = DecomposedWaveform(
                    source,
                    rate,
                    np.full(20, 220),
                    hop_size=120,
                    algorithm="vr",
                    device="cpu",
                )
                second = DecomposedWaveform(
                    source.copy(),
                    rate,
                    np.full(20, 220),
                    hop_size=120,
                    algorithm="vr",
                    device="cpu",
                )
                expected = first.harmonic()
                np.testing.assert_array_equal(second.harmonic(), expected)
                self.assertEqual(len(expected), len(source))
                self.assertEqual(model.calls[-1], 4410)
                np.testing.assert_allclose(
                    expected[30:-30], source[30:-30] * 0.75, atol=2e-3
                )
        self.assertEqual(len(model.calls), 3)

    def test_amplitude_projection_and_unvoiced_mask(self):
        rate = 24000
        f0 = np.full(40, 200, dtype=np.float32)
        uv = np.zeros(40, dtype=bool)
        uv[-1] = True
        source = (0.3 * np.cos(np.arange(4800) * 200 * 2 * np.pi / rate)).astype(
            np.float32
        )
        result = harmonic_amplitudes(source, f0, uv, "cpu")
        np.testing.assert_allclose(result[3:-3, 0], 0.3, atol=1e-5)
        np.testing.assert_allclose(result[3:-3, 1:], 0, atol=1e-5)
        self.assertTrue((result[-1] == 0).all())

    def test_pair_identity_includes_every_transform(self):
        identities = {
            pairing_id("0:sample", key_shift=shift, speed=speed, spk_id=speaker)[0]
            for shift, speed, speaker in (
                (0, 1, None),
                (2, 1, None),
                (0, 1.2, None),
                (2, 1.2, None),
                (2, 1.2, 1),
            )
        }
        self.assertEqual(len(identities), 5)
        self.assertEqual(pairing_id("0:sample"), pairing_id("0:sample", 0, 1, None))

    def test_disabled_orchestration_has_no_vocoder(self):
        from preprocessing.all_in_one_binarizer import AllInOneBinarizer

        hparams["all_in_one"]["mini_ddsp"]["enabled"] = False
        hparams["binary_data_dir"] = self.temporary.name
        with patch(
            "preprocessing.all_in_one_binarizer.AcousticBinarizer"
        ) as acoustic, patch(
            "preprocessing.all_in_one_binarizer.VarianceBinarizer"
        ) as variance:
            AllInOneBinarizer().process()
            acoustic.return_value.process.assert_called_once()
            variance.return_value.process.assert_called_once()
        self.assertFalse((pathlib.Path(self.temporary.name) / "vocoder").exists())


if __name__ == "__main__":
    unittest.main()
