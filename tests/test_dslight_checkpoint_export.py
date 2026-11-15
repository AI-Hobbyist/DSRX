import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from deployment.exporters.acoustic_exporter import DiffSingerAcousticExporter
from modules.toplevel import DiffSingerAcoustic
from utils.hparams import hparams, set_hparams
from utils.lora import extract_lora_state_dict, inject_lora
from utils.phoneme_utils import load_phoneme_dictionary


class DSLightCheckpointExportTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        set_hparams(config='configs/templates/config_acoustic_dslight_quality.yaml', print_hparams=False)
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
        path = self.root / 'model_ckpt_steps_1.ckpt'
        torch.save({'category': 'acoustic', 'state_dict': state}, path)
        return path

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

    def test_dslight_backbone_is_rejected_for_legacy(self):
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        self.write_checkpoint()
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_dslight_aux_is_rejected_for_legacy(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        self.write_checkpoint()
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
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

    def test_legacy_partial_checkpoint_remains_permissive(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        self.write_checkpoint()
        path = self.root / 'model_ckpt_steps_1.ckpt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        checkpoint['state_dict'].pop('model.fs2.txt_embed.weight')
        torch.save(checkpoint, path)
        self.assertEqual(self.export_model().backbone_type, 'lynxnet2')

    def test_lora_loads_matching_dslight_base(self):
        self.write_checkpoint()
        hparams['lora'] = {
            'enabled': True,
            'base_ckpt': str(self.root / 'model_ckpt_steps_1.ckpt'),
            'target_modules': ['linear'],
            'rank': 2,
            'alpha': 2,
            'merge_before_export': True,
        }
        self.assertEqual(self.export_model().backbone_type, 'dslight')

    def test_lora_rejects_wrong_adapter_shape(self):
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        model = DiffSingerAcoustic(vocab_size=len(load_phoneme_dictionary()), out_dims=128)
        inject_lora(model, rank=2, alpha=2, target_modules=['linear'])
        adapter = extract_lora_state_dict(model)
        adapter_key = next(key for key in adapter if key.endswith('.lora_A'))
        adapter[adapter_key] = adapter[adapter_key][:-1]
        torch.save({'category': 'acoustic', 'state_dict': adapter}, self.root / 'model_ckpt_steps_2.ckpt')
        hparams['lora'] = {
            'enabled': True, 'base_ckpt': str(base_path), 'target_modules': ['linear'],
            'rank': 2, 'alpha': 2, 'merge_before_export': True,
        }
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_lora_adapter_only_update_loads(self):
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        model = DiffSingerAcoustic(vocab_size=len(load_phoneme_dictionary()), out_dims=128)
        inject_lora(model, rank=2, alpha=2, target_modules=['linear'])
        adapter = extract_lora_state_dict(model)
        torch.save({'category': 'acoustic', 'state_dict': adapter}, self.root / 'model_ckpt_steps_2.ckpt')
        hparams['lora'] = {
            'enabled': True, 'base_ckpt': str(base_path), 'target_modules': ['linear'],
            'rank': 2, 'alpha': 2, 'merge_before_export': True,
        }
        self.assertEqual(self.export_model().backbone_type, 'dslight')

    def test_legacy_lora_rejects_dslight_backbone_adapter(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        legacy_target = copy.deepcopy(hparams)
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        model = DiffSingerAcoustic(vocab_size=len(load_phoneme_dictionary()), out_dims=128)
        inject_lora(model, rank=2, alpha=2, target_modules=['linear'])
        adapter = extract_lora_state_dict(model)
        torch.save({'category': 'acoustic', 'state_dict': adapter}, self.root / 'model_ckpt_steps_2.ckpt')
        hparams.clear()
        hparams.update(legacy_target)
        hparams['lora'] = {
            'enabled': True, 'base_ckpt': str(base_path), 'target_modules': ['linear'],
            'rank': 2, 'alpha': 2, 'merge_before_export': True,
        }
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_legacy_lora_rejects_dslight_full_update(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        legacy_target = copy.deepcopy(hparams)
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        shutil.copy2(self.write_checkpoint(), self.root / 'model_ckpt_steps_2.ckpt')
        hparams.clear()
        hparams.update(legacy_target)
        hparams['lora'] = {
            'enabled': True, 'base_ckpt': str(base_path), 'target_modules': ['linear'],
            'rank': 2, 'alpha': 2, 'merge_before_export': True,
        }
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_legacy_lora_rejects_dslight_base(self):
        path = self.write_checkpoint()
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        hparams['lora'] = {
            'enabled': True, 'base_ckpt': str(path), 'target_modules': ['linear'],
            'rank': 2, 'alpha': 2, 'merge_before_export': True,
        }
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_lora_rejects_wrong_latest_full_checkpoint(self):
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        self.write_checkpoint()
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        hparams['lora'] = {
            'enabled': True, 'base_ckpt': str(base_path), 'target_modules': ['linear'],
            'rank': 2, 'alpha': 2, 'merge_before_export': True,
        }
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_lora_rejects_wrong_dslight_dimensions(self):
        self.write_checkpoint()
        hparams['backbone_args']['num_channels'] = 128
        hparams['lora'] = {
            'enabled': True,
            'base_ckpt': str(self.root / 'model_ckpt_steps_1.ckpt'),
            'target_modules': ['linear'],
            'rank': 2,
            'alpha': 2,
            'merge_before_export': True,
        }
        with self.assertRaises(RuntimeError):
            self.export_model()

    def test_lora_rejects_wrong_architecture_base(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        self.write_checkpoint()
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        hparams['lora'] = {
            'enabled': True,
            'base_ckpt': str(self.root / 'model_ckpt_steps_1.ckpt'),
            'target_modules': ['linear'],
            'rank': 2,
            'alpha': 2,
            'merge_before_export': True,
        }
        with self.assertRaises(RuntimeError):
            self.export_model()

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
