import unittest
from types import SimpleNamespace

import torch
from torch import nn

from modules.losses import RectifiedFlowLoss
from modules.toplevel import DiffSingerAcoustic
from training.acoustic_task import AcousticTask
from utils.hparams import set_hparams


class AcousticSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_case(self, config):
        h = set_hparams(config=config, print_hparams=False)
        model = DiffSingerAcoustic(vocab_size=10, out_dims=128).eval()
        task = SimpleNamespace(
            model=model, required_variances=[], diffusion_type='reflow',
            lambda_aux_mel_loss=h['lambda_aux_mel_loss'], aux_mel_loss=nn.L1Loss(),
            mel_loss=RectifiedFlowLoss(loss_type=h['main_loss_type'], log_norm=h['main_loss_log_norm']),
        )
        sample = {
            'tokens': torch.tensor([[1, 2, 3, 4]]),
            'mel2ph': torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]]),
            'mel': torch.full((1, 8, 128), -6.0),
            'f0': torch.full((1, 8), 220.0),
            'key_shift': torch.zeros(1, 1), 'speed': torch.ones(1, 1),
            'spk_ids': torch.zeros(1, dtype=torch.long),
            'languages': torch.ones(1, 4, dtype=torch.long),
        }
        return model, task, sample

    def losses(self, task, sample):
        torch.manual_seed(42)
        return AcousticTask.run_model(task, sample)

    def test_dslight_training_and_aux_weight(self):
        torch.manual_seed(123)
        model, task, sample = self.make_case('configs/templates/config_acoustic_dslight.yaml')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        initial = self.losses(task, sample)
        task.lambda_aux_mel_loss *= 2
        doubled = self.losses(task, sample)
        torch.testing.assert_close(doubled['aux_mel_loss'], 2 * initial['aux_mel_loss'])
        task.lambda_aux_mel_loss /= 2
        start = {name: value.item() for name, value in initial.items()}
        for _ in range(20):
            optimizer.zero_grad()
            losses = self.losses(task, sample)
            loss = sum(losses.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        end = {name: value.item() for name, value in self.losses(task, sample).items()}
        print('synthetic fixed-batch losses:', start, '->', end)
        self.assertLess(end['aux_mel_loss'], start['aux_mel_loss'])
        self.assertLess(end['mel_loss'], start['mel_loss'])

    def test_legacy_training_forward_backward(self):
        model, task, sample = self.make_case('configs/original/acoustic.yaml')
        losses = self.losses(task, sample)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
