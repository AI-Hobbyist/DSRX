import copy
from types import SimpleNamespace
import unittest

import torch

from deployment.modules.toplevel import (
    DiffSingerAcousticNative,
    DiffSingerAcousticONNX,
    DiffSingerVarianceNative,
    DiffSingerVarianceONNX,
)
from modules.duration import get_duration_type_config
from modules.toplevel import DiffSingerVariance
from tests.test_aux_dit import small_acoustic_config
from training.variance_task import VarianceTask
from utils.hparams import hparams


class MainMergeCompatibilityTest(unittest.TestCase):
    def test_native_and_onnx_deployment_names(self):
        self.assertIs(DiffSingerAcousticNative, DiffSingerAcousticONNX)
        self.assertIs(DiffSingerVarianceNative, DiffSingerVarianceONNX)

    def test_duration_types_with_shared_task_loss_path(self):
        previous = copy.deepcopy(hparams)
        torch.set_num_threads(2)
        try:
            for mode in ("deterministic", "ddpm", "reflow"):
                with self.subTest(mode=mode):
                    config = small_acoustic_config()
                    config.update(
                        predict_dur=True,
                        predict_pitch=False,
                        use_melody_encoder=False,
                    )
                    for name in ("energy", "breathiness", "voicing", "tension"):
                        config[f"predict_{name}"] = False
                    if mode != "deterministic":
                        config["dur_prediction_args"] = {
                            "type": mode,
                            mode: {
                                "backbone_type": "dslight",
                                "backbone_args": {
                                    "num_channels": 8,
                                    "num_layers": 1,
                                    "cond_channels": 16,
                                    "time_embed_dim": 8,
                                },
                                "timesteps": 4,
                                "sampling_steps": 2,
                                "sampling_algorithm": (
                                    "ddim" if mode == "ddpm" else "euler"
                                ),
                            },
                        }
                    hparams.update(config)
                    model = DiffSingerVariance(16)
                    task = SimpleNamespace(
                        model=model,
                        predict_dur=True,
                        predict_pitch=False,
                        predict_variances=False,
                        variance_prediction_list=[],
                        use_spk_id=False,
                        use_lang_id=False,
                        duration_type=mode,
                        duration_type_hparams=get_duration_type_config(
                            config["dur_prediction_args"], mode
                        ),
                        lambda_dur_loss=1.0,
                        register_validation_loss=lambda *_: None,
                    )
                    VarianceTask.build_losses_and_metrics(task, register_metrics=False)
                    sample = {
                        "tokens": torch.tensor([[1, 2, 3]]),
                        "ph_dur": torch.tensor([[4, 4, 4]]),
                        "ph2word": torch.tensor([[1, 1, 2]]),
                        "midi": torch.tensor([[60, 60, 62]]),
                        "size": 1,
                    }
                    loss = VarianceTask.run_model(task, sample, modules={"dur"})[
                        "dur_loss"
                    ]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    model.eval()
                    with torch.no_grad():
                        prediction, _, _ = VarianceTask.run_model(
                            task, sample, infer=True, dur_seed=1234
                        )
                    self.assertEqual(prediction.shape, (1, 3))
                    if mode != "deterministic":
                        self.assertEqual(prediction[0, :2].sum().item(), 8)
                        self.assertEqual(prediction[0, 2].item(), 4)
        finally:
            hparams.clear()
            hparams.update(previous)


if __name__ == "__main__":
    unittest.main()
