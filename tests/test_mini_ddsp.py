import copy
import unittest
from unittest.mock import patch

import torch

from modules.vocoders.mini_ddsp import MiniDDSP, mini_ddsp_config


def config(sample_rate=44100):
    return {
        "audio_sample_rate": sample_rate,
        "hop_size": 16,
        "audio_num_mel_bins": 8,
        "all_in_one": {
            "enabled": True,
            "mini_ddsp": {
                "enabled": True,
                "harmonics": 7,
                "noise_bands": 4,
                "hidden_size": 16,
                "num_layers": 2,
                "kernel_size": 3,
                "harmonic_group_size": 3,
            },
        },
    }


class MiniDDSPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_rates_batches_lengths_gradients_and_resources(self):
        for rate in (24000, 44100, 48000):
            model = MiniDDSP(config(rate))
            for frames in (1, 4, 5):
                with self.subTest(rate=rate, frames=frames):
                    mel = torch.randn(2, frames, 8, requires_grad=True)
                    f0 = torch.full((2, frames), 220.0)
                    f0[1] = 0
                    lengths = torch.tensor([frames, max(frames - 1, 0)])
                    result = model(mel, f0, lengths, return_components=True)
                    self.assertEqual(result["waveform"].shape, (2, frames * 16))
                    self.assertTrue(torch.isfinite(result["waveform"]).all())
                    self.assertEqual(result["harmonic"][1].abs().max(), 0)
                    self.assertEqual(
                        result["waveform"][1, lengths[1] * 16 :].abs().max(), 0
                    )
                    result["waveform"].square().mean().backward()
                    self.assertTrue(torch.isfinite(mel.grad).all())
                    self.assertGreater(mel.grad.abs().sum(), 0)
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    model.zero_grad()
            restored = MiniDDSP(config(rate))
            restored.load_state_dict(model.state_dict(), strict=True)
            torch.testing.assert_close(
                model(mel, f0), restored(mel, f0), rtol=0, atol=0
            )
            self.assertEqual(model.noise_basis.shape, (4, rate))

    def test_grouped_matches_direct_waveform_and_gradients(self):
        model = MiniDDSP(config())
        for frames in (1, 4, 5):
            amplitude = torch.rand(2, frames, 7, requires_grad=True)
            phase = torch.rand(2, frames, 7, requires_grad=True)
            frequencies = torch.rand(2, frames, 7) * 4000
            grouped = model.synthesize_harmonics(amplitude, phase, frequencies)
            direct = model.synthesize_harmonics(
                amplitude, phase, frequencies, group_size=7
            )
            torch.testing.assert_close(grouped, direct, rtol=2e-5, atol=2e-6)
            grad_group = torch.autograd.grad(grouped.square().sum(), (amplitude, phase))
            grad_direct = torch.autograd.grad(direct.square().sum(), (amplitude, phase))
            for actual, expected in zip(grad_group, grad_direct):
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_overlap_add_matches_explicit_frame_placement(self):
        model = MiniDDSP(config())
        for frames in (1, 4, 5):
            windows = torch.randn(2, frames, 32)
            reference = torch.zeros(2, (frames + 1) * 16)
            for index in range(frames):
                reference[:, index * 16 : index * 16 + 32] += windows[:, index]
            torch.testing.assert_close(model._overlap_add(windows), reference[:, 16:])

    def test_default_disabled_and_invalid_scope(self):
        self.assertFalse(mini_ddsp_config({}).get("enabled", False))
        invalid = config()
        invalid["all_in_one"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "all_in_one"):
            mini_ddsp_config(invalid)
        from modules.toplevel import DiffSingerAllInOne
        from utils.hparams import hparams

        saved = copy.deepcopy(hparams)
        try:
            for enabled in (False, True):
                current = config()
                current["all_in_one"]["mini_ddsp"]["enabled"] = enabled
                hparams.clear()
                hparams.update(current)
                with patch(
                    "modules.toplevel.DiffSingerAcoustic",
                    return_value=torch.nn.Linear(1, 1),
                ), patch(
                    "modules.toplevel.DiffSingerVariance",
                    return_value=torch.nn.Linear(1, 1),
                ):
                    model = DiffSingerAllInOne(10, 8)
                self.assertEqual(hasattr(model, "vocoder"), enabled)
                self.assertEqual(
                    any(k.startswith("vocoder.") for k in model.state_dict()), enabled
                )
        finally:
            hparams.clear()
            hparams.update(saved)


if __name__ == "__main__":
    unittest.main()
