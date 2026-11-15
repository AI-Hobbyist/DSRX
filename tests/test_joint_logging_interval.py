import unittest
import tempfile
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from basics.base_task import BaseTask
from training.all_in_one_task import AllInOneTask
from utils.training_utils import DsTensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class JointLoggingIntervalTests(unittest.TestCase):
    def make_task(self):
        return SimpleNamespace(
            mini_ddsp_settings={"enabled": True},
            global_step=1,
            _joint_log_sums={},
            _joint_log_count=0,
            _joint_log_names=["ddsp_joint_loss", "ddsp_aux_joint_loss"],
        )

    def collect(self, task, outputs):
        with patch("training.all_in_one_task.hparams", {"log_interval": 5}):
            return AllInOneTask._training_log_metrics(task, outputs)

    def test_valid_samples_between_logging_steps_are_included(self):
        task = self.make_task()
        for step, value, count in [(1, 2.0, 1), (2, 4.0, 3)]:
            task.global_step = step
            self.collect(
                task,
                {
                    "ddsp_joint_loss": torch.tensor(value, requires_grad=True),
                    "ddsp_aux_joint_loss": torch.tensor(value + 1),
                    "ddsp_joint_sample_count": count,
                },
            )
        task.global_step = 5
        result = self.collect(task, {"ddsp_joint_sample_count": 0, "mel_loss": 0.1})
        self.assertEqual(result["ddsp_joint_loss"], 3.5)
        self.assertEqual(result["ddsp_aux_joint_loss"], 4.5)
        self.assertNotIn("ddsp_joint_sample_count", result)
        self.assertFalse(result["ddsp_joint_loss"].requires_grad)
        self.assertEqual(result["mel_loss"], 0.1)
        self.assertEqual(task._joint_log_count, 0)
        self.assertEqual(task._joint_log_sums, {})

    def test_empty_interval_marks_missing_measurement_without_reusing_old_value(self):
        task = self.make_task()
        task.global_step = 5
        result = self.collect(task, {"ddsp_joint_sample_count": 0})
        self.assertNotIn("ddsp_joint_loss", result)
        self.assertNotIn("ddsp_aux_joint_loss", result)
        self.assertNotIn("ddsp_joint_sample_count", result)

    def test_base_task_collects_between_log_steps_and_writes_at_original_interval(self):
        task = SimpleNamespace(
            global_step=1,
            current_epoch=0,
            log=Mock(),
            log_dict=Mock(),
            logger=SimpleNamespace(log_metrics=Mock()),
            lr_schedulers=lambda: SimpleNamespace(get_last_lr=lambda: [0.001]),
            _training_step=Mock(return_value=(torch.tensor(1.0), {"mel_loss": 0.2})),
            _training_log_metrics=Mock(
                return_value={"mel_loss": 0.2, "ddsp_joint_loss": 3.0}
            ),
        )
        with patch(
            "basics.base_task.hparams", {"log_interval": 5, "val_check_interval": "1ep"}
        ):
            BaseTask.training_step(task, {}, 0)
            task.logger.log_metrics.assert_not_called()
            task.global_step = 5
            BaseTask.training_step(task, {}, 1)
        self.assertEqual(task._training_log_metrics.call_count, 2)
        recorded = task.logger.log_metrics.call_args
        self.assertEqual(recorded.kwargs["step"], 5)
        self.assertEqual(recorded.args[0]["training/ddsp_joint_loss"], 3.0)

    def test_real_tensorboard_receives_interval_value_when_logging_batch_is_unpaired(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            task = self.make_task()
            task.current_epoch = 0
            task.log = Mock()
            task.log_dict = Mock()
            task.logger = DsTensorBoardLogger(directory, version="latest")
            task.lr_schedulers = lambda: SimpleNamespace(get_last_lr=lambda: [0.001])
            task._training_log_metrics = MethodType(
                AllInOneTask._training_log_metrics, task
            )
            task._training_step = Mock(
                side_effect=[
                    (
                        torch.tensor(1.0),
                        {
                            "ddsp_joint_loss": torch.tensor(2.0),
                            "ddsp_aux_joint_loss": torch.tensor(3.0),
                            "ddsp_joint_sample_count": 2,
                        },
                    ),
                    (torch.tensor(1.0), {"ddsp_joint_sample_count": 0}),
                ]
            )
            settings = {"log_interval": 5, "val_check_interval": "1ep"}
            with patch("training.all_in_one_task.hparams", settings), patch(
                "basics.base_task.hparams", settings
            ):
                BaseTask.training_step(task, {}, 0)
                task.global_step = 5
                BaseTask.training_step(task, {}, 1)
            task.logger.experiment.flush()
            task.logger.experiment.close()
            events = EventAccumulator(task.logger.log_dir).Reload()
            self.assertNotIn("training/ddsp_joint_sample_count", events.Tags()["scalars"])
            for tag, value in [("ddsp_joint_loss", 2.0), ("ddsp_aux_joint_loss", 3.0)]:
                points = events.Scalars("training/" + tag)
                self.assertEqual(
                    [(point.step, point.value) for point in points], [(5, value)]
                )


if __name__ == "__main__":
    unittest.main()
