import copy
import tempfile
import unittest
from pathlib import Path

import torch

from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter
from modules.toplevel import DiffSingerAcoustic
from utils.hparams import hparams, set_hparams
from utils.phoneme_utils import load_phoneme_dictionary


class DSLightCheckpointExportTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        set_hparams(config='configs/dslight/template.yaml', print_hparams=False)
        hparams['work_dir'] = str(self.root)
        hparams['exp_name'] = 'checkpoint_test'
        self.light_config = copy.deepcopy(hparams)
        self.legacy_config = set_hparams(
            config='configs/original/acoustic.yaml', print_hparams=False, global_hparams=False,
        )

    def tearDown(self):
        self.directory.cleanup()

    def write_checkpoint(self):
        model = DiffSingerAcoustic(vocab_size=len(load_phoneme_dictionary()), out_dims=128)
        state = {'model.' + key: value for key, value in model.state_dict().items()}
        torch.save({'category': 'acoustic', 'state_dict': state}, self.root / 'model_ckpt_steps_1.ckpt')

    def export_model(self):
        return DiffSingerAcousticExporter(device='cpu', cache_dir=self.root / 'cache').model

    def test_matching_dslight_checkpoint_loads(self):
        self.write_checkpoint()
        self.assertEqual(self.export_model().backbone_type, 'dslight')

    def test_legacy_backbone_is_rejected_for_dslight(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        self.write_checkpoint()
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_aux_only_dslight_rejects_legacy_aux_checkpoint(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        self.write_checkpoint()
        hparams['shallow_diffusion_args'] = copy.deepcopy(self.light_config['shallow_diffusion_args'])
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_legacy_configuration_still_loads(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        self.write_checkpoint()
        self.assertEqual(self.export_model().backbone_type, 'lynxnet2')

    def test_legacy_aux_decoder_is_rejected_for_dslight(self):
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        self.write_checkpoint()
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        with self.assertRaises(RuntimeError):
            self.export_model()


if __name__ == '__main__':
    unittest.main()
