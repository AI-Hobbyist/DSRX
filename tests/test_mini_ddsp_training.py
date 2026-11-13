import unittest
from unittest.mock import patch

import torch
from torch import nn

from modules.core.ddpm import GaussianDiffusion
from modules.core.reflow import RectifiedFlow
from training.all_in_one_task import AllInOneTask
from training.mini_ddsp import joint_factor


class MiniDDSPTrainingTests(unittest.TestCase):
    def test_reflow_clean_estimate_uses_the_training_velocity(self):
        flow = RectifiedFlow.__new__(RectifiedFlow)
        nn.Module.__init__(flow)
        flow.time_scale_factor = 1000
        target = torch.randn(2, 1, 8, 5)
        time = torch.tensor([0.2, 0.7])
        torch.manual_seed(123)
        noise = torch.randn_like(target)
        velocity = (target - noise).requires_grad_()
        torch.manual_seed(123)
        with patch.object(flow, "_run_velocity", return_value=velocity):
            predicted, _, clean = flow.p_losses(target, time, None, return_clean=True)
        torch.testing.assert_close(clean, target)
        clean.sum().backward()
        torch.testing.assert_close(
            velocity.grad, (1 - time[:, None, None, None]).expand_as(target)
        )
        self.assertIs(predicted, velocity)

    def test_ddpm_clean_estimate_uses_alpha_parameterization(self):
        diffusion = GaussianDiffusion.__new__(GaussianDiffusion)
        nn.Module.__init__(diffusion)
        alpha = torch.tensor([0.99, 0.5, 0.1])
        diffusion.register_buffer("sqrt_alphas_cumprod", alpha.sqrt())
        diffusion.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alpha).sqrt())
        diffusion.register_buffer("sqrt_recip_alphas_cumprod", alpha.rsqrt())
        diffusion.register_buffer("sqrt_recipm1_alphas_cumprod", (1 / alpha - 1).sqrt())
        target = torch.randn(2, 1, 8, 5)
        noise = torch.randn_like(target, requires_grad=True)
        time = torch.tensor([0, 2])
        with patch.object(diffusion, "_run_denoise", return_value=noise):
            _, _, clean = diffusion.p_losses(
                target, time, None, noise=noise, return_clean=True
            )
        torch.testing.assert_close(clean, target, rtol=1e-5, atol=1e-6)
        clean.square().sum().backward()
        self.assertTrue(torch.isfinite(noise.grad).all())
        self.assertGreater(noise.grad.abs().sum(), 0)

    def test_schedule_and_two_stream_budget(self):
        settings = {"warmup_steps": 2, "joint_ramp_steps": 4}
        self.assertEqual(
            [joint_factor(step, settings) for step in (0, 1, 2, 3, 4, 6, 9)],
            [0, 0, 0, 0.25, 0.5, 1, 1],
        )
        self.assertEqual(joint_factor(2, {**settings, "joint_ramp_steps": 0}), 1)
        self.assertEqual(AllInOneTask.training_stream_count, 2)
        self.assertEqual(
            AllInOneTask._split_training_batch_limit(8, "max_batch_size"), 4
        )

    def test_initialization_allows_only_complete_new_vocoder(self):
        class Harness(nn.Module):
            def __init__(self):
                super().__init__()
                self.mini_ddsp_settings = {"enabled": True}
                self.model = nn.Module()
                self.model.acoustic = nn.Linear(2, 2)
                self.model.vocoder = nn.Linear(2, 2)

        model = Harness()
        old = {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("model.vocoder.")
        }
        AllInOneTask.load_finetune_ckpt(model, old)
        broken = dict(old)
        broken.pop("model.acoustic.weight")
        with self.assertRaisesRegex(RuntimeError, "only permits"):
            AllInOneTask.load_finetune_ckpt(model, broken)
        partial = dict(old)
        partial["model.vocoder.weight"] = model.model.vocoder.weight
        with self.assertRaisesRegex(RuntimeError, "only permits"):
            AllInOneTask.load_finetune_ckpt(model, partial)


if __name__ == "__main__":
    unittest.main()
