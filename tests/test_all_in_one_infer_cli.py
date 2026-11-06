import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from click.testing import CliRunner

from modules.toplevel import DiffSingerAllInOne
from scripts import infer
from utils.hparams import hparams, set_hparams


class AllInOneInferCliTest(unittest.TestCase):
    def test_subcommand_help(self):
        runner = CliRunner()
        for command in ('all-in-one', 'acoustic', 'variance'):
            result = runner.invoke(infer.main, [command, '--help'])
            self.assertEqual(result.exit_code, 0, result.output)

    def test_loads_full_checkpoint_and_passes_variance_output_to_acoustic(self):
        torch.set_num_threads(1)
        config = set_hparams(
            config='configs/templates/all_in_one_dslight_quality_50mb.yaml',
            print_hparams=False,
        ).copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / 'ckpt' / 'test'
            experiment.mkdir(parents=True)
            config['work_dir'] = str(experiment)
            model = DiffSingerAllInOne(vocab_size=4, out_dims=128)
            with torch.no_grad():
                next(model.acoustic.parameters()).fill_(0.125)
                next(model.variance.parameters()).fill_(0.25)
            torch.save(
                {
                    'state_dict': {f'model.{key}': value for key, value in model.state_dict().items()},
                    'category': 'all_in_one',
                },
                experiment / 'model_ckpt_steps_42.ckpt',
            )
            source = root / 'input.ds'
            source.write_text(json.dumps({'ph_seq': 'a'}), encoding='utf-8')
            output = root / 'output'
            predicted = [{'ph_seq': 'a', 'ph_dur': '0.1', 'f0_seq': '440'}]
            calls = {}

            class VarianceInfer:
                def __init__(self, **kwargs):
                    calls['variance'] = kwargs

                def run_inference(self, params, *, out_dir, title, seed):
                    calls['params'] = params
                    (out_dir / f'{title}.ds').write_text(json.dumps(predicted), encoding='utf-8')

            class AcousticInfer:
                def __init__(self, **kwargs):
                    calls['acoustic'] = kwargs

                def run_inference(self, params, **kwargs):
                    calls['acoustic_params'] = params
                    calls['output'] = kwargs

            def load_config():
                hparams.clear()
                hparams.update(config)

            hparams_module = importlib.import_module('utils.hparams')
            with patch.object(infer, 'root_dir', root), \
                    patch.object(hparams_module, 'set_hparams', side_effect=load_config), \
                    patch('utils.phoneme_utils.load_phoneme_dictionary', return_value=range(4)), \
                    patch('torch.cuda.is_available', return_value=False), \
                    patch('inference.ds_variance.DiffSingerVarianceInfer', VarianceInfer), \
                    patch('inference.ds_acoustic.DiffSingerAcousticInfer', AcousticInfer):
                result = CliRunner().invoke(infer.main, [
                    'all-in-one', str(source), '--exp', 'test', '--ckpt', '42',
                    '--out', str(output), '--title', 'result', '--steps', '10',
                    '--vocoder-fp32',
                ])

            self.assertEqual(result.exit_code, 0, str(result.exception) + result.output)
            self.assertTrue(torch.all(next(calls['variance']['model'].parameters()) == 0.25))
            self.assertTrue(torch.all(next(calls['acoustic']['model'].parameters()) == 0.125))
            self.assertEqual(calls['acoustic_params'], predicted)
            self.assertEqual(calls['params'][0]['lang'], 'zh')
            self.assertIn('dur', calls['variance']['predictions'])
            self.assertEqual(calls['output']['title'], 'result')
            self.assertEqual(hparams['sampling_steps'], 10)
            self.assertFalse(hparams['inference_vocoder_nsf_selective_fp16'])
            self.assertTrue((output / 'result_variance.ds').is_file())


if __name__ == '__main__':
    unittest.main()
