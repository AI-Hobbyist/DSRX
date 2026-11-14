import unittest

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

from utils.training_utils import get_strategy


class LightningStrategyTests(unittest.TestCase):
    def test_automatic_cpu_strategy(self):
        strategy = get_strategy(devices=1, accelerator="cpu", precision="32-true")
        self.assertIsInstance(strategy, SingleDeviceStrategy)
        self.assertEqual(strategy.root_device, torch.device("cpu"))
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            strategy=strategy,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        self.assertEqual(trainer.strategy.root_device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_automatic_cuda_strategy(self):
        strategy = get_strategy(devices=[0], accelerator="gpu", precision="bf16-mixed")
        self.assertIsInstance(strategy, SingleDeviceStrategy)
        self.assertEqual(strategy.root_device, torch.device("cuda", 0))
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
            strategy=strategy,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        self.assertEqual(trainer.strategy.root_device, torch.device("cuda", 0))

    def test_explicit_distributed_options_are_preserved(self):
        strategy = get_strategy(
            devices=1,
            accelerator="cpu",
            precision="32-true",
            strategy={
                "name": "ddp",
                "process_group_backend": "gloo",
                "find_unused_parameters": True,
            },
        )
        self.assertIsInstance(strategy, DDPStrategy)
        self.assertEqual(strategy._process_group_backend, "gloo")
        self.assertTrue(strategy._ddp_kwargs["find_unused_parameters"])


if __name__ == "__main__":
    unittest.main()
