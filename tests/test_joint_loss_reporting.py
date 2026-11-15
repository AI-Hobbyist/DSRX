import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from modules.vocoders.mini_ddsp import MiniDDSP
from tests.test_mini_ddsp import config
from training.mini_ddsp import MiniDDSPLoss
from training.all_in_one_task import AllInOneTask


class JointLossReportingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        settings = config()
        settings["win_size"] = 32
        settings["all_in_one"]["mini_ddsp"].update(
            warmup_steps=2, joint_ramp_steps=4, segment_frames=5
        )
        self.vocoder = MiniDDSP(settings)
        self.loss = MiniDDSPLoss(settings)
        mel = torch.randn(1, 12, 8) - 4
        f0 = torch.full((1, 12), 220.0)
        with torch.no_grad():
            components = self.vocoder(mel, f0, return_components=True)
        self.sample = {
            "size": 1,
            "mel": mel,
            "f0": f0,
            "vocoder": {
                "mel": mel,
                "f0": f0,
                "paired": torch.tensor([True]),
                "frame_lengths": torch.tensor([12]),
                "sample_lengths": torch.tensor([192]),
                "waveform": components["waveform"],
                "amplitude": components["amplitude"],
            },
        }
        self.output = SimpleNamespace(
            clean_out=(mel + 0.5).requires_grad_(), aux_out=None
        )

    def test_training_raw_metric_is_independent_of_ramp_and_not_in_objective(self):
        half, half_metrics = self.loss(
            self.vocoder,
            self.sample,
            self.output,
            None,
            4,
            training=False,
            return_metrics=True,
        )
        full, full_metrics = self.loss(
            self.vocoder,
            self.sample,
            self.output,
            None,
            6,
            training=False,
            return_metrics=True,
        )
        torch.testing.assert_close(full["ddsp_joint_loss"], half["ddsp_joint_loss"] * 2)
        torch.testing.assert_close(
            full_metrics["ddsp_joint_raw_loss"], half_metrics["ddsp_joint_raw_loss"]
        )
        self.assertNotIn("ddsp_joint_raw_loss", full)
        full["ddsp_joint_loss"].backward()
        self.assertGreater(self.output.clean_out.grad.abs().sum(), 0)

    def test_missing_joint_samples_do_not_create_zero_raw_measurement(self):
        self.sample["vocoder"]["paired"][0] = False
        losses, metrics = self.loss(
            self.vocoder,
            self.sample,
            self.output,
            None,
            6,
            training=False,
            return_metrics=True,
        )
        self.assertEqual(losses["ddsp_joint_loss"], 0)
        self.assertEqual(metrics["ddsp_joint_sample_count"], 0)
        self.assertNotIn("ddsp_joint_raw_loss", metrics)

    def test_inference_metric_uses_full_waveform_and_no_schedule_or_loss_weight(self):
        output = SimpleNamespace(
            diff_out=self.sample["mel"], aux_out=self.sample["mel"]
        )
        result = self.loss.validation_joint(self.vocoder, self.sample, output)
        self.assertLess(result["ddsp_joint_loss"], 1e-6)
        self.assertLess(result["ddsp_aux_joint_loss"], 1e-6)
        self.loss.settings["loss"] = {"joint_weight": 100}
        changed = self.loss.validation_joint(self.vocoder, self.sample, output)
        torch.testing.assert_close(result, changed)

    def test_validation_noise_is_repeatable_and_does_not_change_training_rng(self):
        task = SimpleNamespace(
            device=torch.device("cpu"),
            model=SimpleNamespace(acoustic=None, vocoder=None),
            ddsp_loss=SimpleNamespace(
                validation_joint=lambda vocoder, sample, output: output
            ),
        )
        with patch(
            "training.all_in_one_task.AcousticTask.run_model",
            side_effect=lambda *args, **kwargs: torch.randn(3),
        ) as run:
            state = torch.random.get_rng_state().clone()
            first = AllInOneTask._joint_validation(task, {}, 0)
            torch.testing.assert_close(torch.random.get_rng_state(), state)
            torch.randn(11)
            second = AllInOneTask._joint_validation(task, {}, 0)
            torch.testing.assert_close(first, second)
            self.assertTrue(run.call_args.kwargs["infer"])
            self.assertIsNone(run.call_args.args[1]["mel"])

    def test_training_logs_raw_loss_but_optimizes_weighted_loss(self):
        task = SimpleNamespace(
            mini_ddsp_settings={"enabled": True},
            global_step=4,
            model=SimpleNamespace(acoustic=None, variance=None, vocoder=None),
            ddsp_loss=Mock(
                return_value=(
                    {"ddsp_joint_loss": torch.tensor(3.0)},
                    {
                        "ddsp_joint_raw_loss": torch.tensor(99.0),
                        "ddsp_joint_factor": 0.5,
                    },
                )
            ),
        )
        with patch(
            "training.all_in_one_task.AcousticTask.run_model",
            return_value=({"mel_loss": torch.tensor(2.0)}, None),
        ), patch(
            "training.all_in_one_task.VarianceTask.run_model",
            return_value={"pitch_loss": torch.tensor(5.0)},
        ):
            total, logs = AllInOneTask._training_step(
                task, {"acoustic": {"size": 1}, "variance": {"size": 1}}
            )
        self.assertEqual(total, 10)
        self.assertEqual(logs["ddsp_joint_loss"], 99)
        self.assertNotIn("ddsp_joint_raw_loss", logs)

    def test_inference_metric_is_not_added_to_validation_total(self):
        names = ["total_loss", "mel_loss", "ddsp_spectral_loss", "ddsp_joint_loss"]
        task = SimpleNamespace(
            device=torch.device("cpu"),
            acoustic_valid_dataset=None,
            mini_ddsp_settings={"enabled": True},
            global_step=0,
            use_vocoder=False,
            model=SimpleNamespace(acoustic=None, vocoder=None),
            _run_validation_step=Mock(
                return_value=({"mel_loss": torch.tensor(2.0)}, 1)
            ),
            ddsp_loss=Mock(return_value={"ddsp_spectral_loss": torch.tensor(3.0)}),
            _joint_validation=Mock(
                return_value={"ddsp_joint_loss": torch.tensor(99.0)}
            ),
            valid_losses={name: Mock() for name in names},
        )
        AllInOneTask.validation_step(task, {"size": 1}, 0)
        self.assertEqual(task.valid_losses["total_loss"].update.call_args.args[0], 5)
        self.assertEqual(
            task.valid_losses["ddsp_joint_loss"].update.call_args.args[0], 99
        )
        self.assertFalse(task.ddsp_loss.call_args.kwargs["include_joint"])


if __name__ == "__main__":
    unittest.main()
