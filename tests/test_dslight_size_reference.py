import tempfile
import unittest
from pathlib import Path

import torch

from modules.toplevel import DiffSingerAllInOne
from utils.hparams import set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


class AllInOneSizeReferenceTest(unittest.TestCase):
    def test_full_model_forward_and_backward(self):
        torch.set_num_threads(1)
        set_hparams(config='configs/dslight/all_in_one_50mb.yaml', print_hparams=False)
        model = DiffSingerAllInOne(vocab_size=len(load_phoneme_dictionary()), out_dims=128)
        self.assertLess(parameter_count(model) * 4, 50_000_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pth'
            torch.save(model.state_dict(), path)
            self.assertLess(path.stat().st_size, 50_000_000)
        self.assertLess(parameter_count(model.variance), parameter_count(model.acoustic))

        tokens = torch.tensor([[1, 2, 3, 4]])
        alignment = torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]])
        curve = torch.zeros(1, 8)
        acoustic = model.acoustic(
            tokens, mel2ph=alignment, f0=torch.full((1, 8), 220.0),
            key_shift=torch.zeros(1, 1), speed=torch.ones(1, 1),
            energy=curve, breathiness=curve, voicing=curve, tension=curve,
            gt_mel=torch.full((1, 8, 128), -6.0), infer=False,
        )
        self.assertEqual(tuple(acoustic.aux_out.shape), (1, 8, 128))
        acoustic_loss = acoustic.aux_out.square().mean() + acoustic.diff_out[0].square().mean()
        self.assertTrue(torch.isfinite(acoustic_loss))
        acoustic_loss.backward()

        duration, pitch, variances = model.variance(
            tokens, midi=torch.full((1, 4), 60), ph2word=torch.tensor([[1, 1, 2, 2]]),
            ph_dur=torch.full((1, 4), 2), mel2ph=alignment,
            base_pitch=torch.full((1, 8), 60.0), pitch=torch.full((1, 8), 60.0),
            energy=curve, breathiness=curve, voicing=curve, tension=curve, infer=False,
        )
        self.assertIsNotNone(duration)
        self.assertIsNotNone(pitch)
        self.assertIsNotNone(variances)
        variance_loss = duration.square().mean() + pitch[0].square().mean() + variances[0].square().mean()
        self.assertTrue(torch.isfinite(variance_loss))
        variance_loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
