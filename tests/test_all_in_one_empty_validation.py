import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from training.all_in_one_task import AllInOneTask


class EmptyVarianceValidationTest(unittest.TestCase):
    def test_zero_predicted_duration_skips_only_optional_audio(self):
        variance_infer = SimpleNamespace(
            preprocess_input=Mock(return_value={}),
            forward_model=Mock(return_value=(
                torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 1), {}
            )),
        )
        acoustic_infer = SimpleNamespace(
            preprocess_input=Mock(return_value={}),
            forward_model=Mock(
                side_effect=lambda _: torch.nn.Conv1d(1, 1, 1)(torch.empty(1, 1, 0))
            ),
        )
        task = SimpleNamespace(
            val_with_variance_enabled=True,
            global_rank=0,
            variance_validation_speakers=[('opencpop', 0)],
            variance_validation_language=None,
            val_with_variance={'enable': True},
            variance_validation_infer=variance_infer,
            acoustic_validation_infer=acoustic_infer,
        )
        config = {
            'dictionaries': {'zh': 'unused'},
            'hop_size': 512,
            'audio_sample_rate': 44100,
            'use_spk_id': False,
        }
        with patch('training.all_in_one_task.hparams', config), \
                patch('training.all_in_one_task.load_validation_sources', return_value=('zh', {})):
            with self.assertWarnsRegex(UserWarning, 'zero.*frames'):
                AllInOneTask._on_validation_epoch_end(task)
        acoustic_infer.preprocess_input.assert_not_called()


if __name__ == '__main__':
    unittest.main()
