import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from training.all_in_one_task import AllInOneTask


class EmptyVarianceValidationTest(unittest.TestCase):
    def make_task(self, predictions):
        acoustic_infer = SimpleNamespace(
            preprocess_input=Mock(return_value={'f0': torch.ones(1, 1)}),
            forward_model=Mock(return_value=torch.ones(1, 1, 1)),
        )
        task = SimpleNamespace(
            val_with_variance_enabled=True,
            global_rank=0,
            global_step=0,
            variance_validation_speakers=[('opencpop', 0)],
            variance_validation_language=None,
            val_with_variance={'enable': True},
            variance_validation_infer=SimpleNamespace(
                preprocess_input=Mock(return_value={}),
                forward_model=Mock(side_effect=predictions),
            ),
            acoustic_validation_infer=acoustic_infer,
            vocoder=SimpleNamespace(spec2wav_torch=Mock(return_value=torch.ones(1))),
            logger=SimpleNamespace(all_rank_experiment=SimpleNamespace(
                add_audio=Mock(), add_figure=Mock(),
            )),
        )
        return task

    def run_validation(self, task, sources):
        config = {
            'dictionaries': {'zh': 'unused'},
            'hop_size': 512,
            'audio_sample_rate': 44100,
            'use_spk_id': False,
            'mel_vmin': -12,
            'mel_vmax': 2,
        }
        with patch('training.all_in_one_task.hparams', config), \
                patch('training.all_in_one_task.iter_validation_sources', return_value=iter(sources)), \
                patch('training.all_in_one_task.spec_to_figure', return_value=None), \
                patch('training.all_in_one_task.plt.close'):
            AllInOneTask._on_validation_epoch_end(task)

    def test_zero_duration_retries_and_produces_audio(self):
        zero = (torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 1), {})
        valid = (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2), {})
        task = self.make_task([zero, valid])
        self.run_validation(task, [('zh', {'text': 'first'}), ('zh', {'text': 'second'})])
        self.assertEqual(task.variance_validation_infer.forward_model.call_count, 2)
        self.assertEqual(task.acoustic_validation_infer.preprocess_input.call_count, 1)
        task.logger.all_rank_experiment.add_audio.assert_called_once()

    def test_exhausted_sources_warn_and_stop(self):
        zero = (torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 1), {})
        task = self.make_task([zero, zero])
        with self.assertWarnsRegex(UserWarning, 'all validation segments.*zero frames'):
            self.run_validation(task, [('zh', {}), ('zh', {})])
        self.assertEqual(task.variance_validation_infer.forward_model.call_count, 2)
        task.acoustic_validation_infer.preprocess_input.assert_not_called()

    def test_zero_pitch_retries(self):
        empty_pitch = (torch.ones(1, 2, dtype=torch.long), torch.empty(1, 0), {})
        valid = (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2), {})
        task = self.make_task([empty_pitch, valid])
        self.run_validation(task, [('zh', {}), ('zh', {})])
        task.logger.all_rank_experiment.add_audio.assert_called_once()


if __name__ == '__main__':
    unittest.main()
