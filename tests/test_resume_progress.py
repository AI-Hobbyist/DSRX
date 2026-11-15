import io
import tempfile
import unittest
from pathlib import Path

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.training_utils import DsTQDMProgressBar


class ProgressOnlyTask(LightningModule):
    """Exercise framework checkpoint/resume without model training or updates."""

    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return None

    def train_dataloader(self):
        loader = DataLoader(TensorDataset(torch.arange(6)), batch_size=1)
        return CombinedLoader(
            {"acoustic": loader, "variance": loader}, "max_size_cycle"
        )


class CaptureProgress(DsTQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.stream = io.StringIO()
        self.snapshots = []
        self.epochs = []

    def init_train_tqdm(self):
        return tqdm(file=self.stream, bar_format=self.BAR_FORMAT, mininterval=0)

    def get_metrics(self, trainer, model):
        return {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        bar = self.train_progress_bar
        self.snapshots.append((batch_idx, bar.n, bar.total, bar.initial, str(bar)))
        self.epochs.append(trainer.current_epoch)


class SaveMidEpoch(Callback):
    def __init__(self, path):
        self.path = path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 1:
            trainer.save_checkpoint(self.path)
            trainer.should_stop = True


def make_trainer(callbacks, max_epochs=1):
    return Trainer(
        accelerator="cpu",
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )


class ResumeProgressTests(unittest.TestCase):
    def test_mid_epoch_resume_displays_total_offset_and_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "mid.ckpt")
            first = CaptureProgress()
            make_trainer([first, SaveMidEpoch(checkpoint)]).fit(ProgressOnlyTask())
            resumed = CaptureProgress()
            make_trainer([resumed], max_epochs=2).fit(
                ProgressOnlyTask(), ckpt_path=checkpoint
            )
        index, completed, total, initial, rendered = resumed.snapshots[0]
        self.assertGreater(index, 0)
        self.assertEqual(total, 6)
        self.assertEqual(initial, index)
        self.assertEqual(completed, index + 1)
        self.assertIn("/6", rendered)
        self.assertNotIn("?", rendered)
        self.assertIn("%", rendered)
        self.assertIn("<", rendered)
        self.assertEqual(resumed.snapshots[-1][1], 6)
        next_epoch = [
            snapshot
            for epoch, snapshot in zip(resumed.epochs, resumed.snapshots)
            if epoch == 1
        ]
        self.assertEqual(next_epoch[0][1:4], (1, 6, 0))
        self.assertEqual(next_epoch[-1][1:3], (6, 6))

    def test_fresh_epoch_starts_at_zero_and_keeps_finite_total(self):
        progress = CaptureProgress()
        make_trainer([progress]).fit(ProgressOnlyTask())
        self.assertEqual(progress.snapshots[0][1:4], (1, 6, 0))
        self.assertEqual(progress.snapshots[-1][1:3], (6, 6))


if __name__ == "__main__":
    unittest.main()
