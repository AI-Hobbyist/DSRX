import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from inference.ds_acoustic import DiffSingerAcousticInfer
from modules.toplevel import DiffSingerAcoustic
from utils.hparams import hparams, set_hparams
from utils.lora import extract_lora_state_dict, inject_lora
from utils.phoneme_utils import load_phoneme_dictionary


class DSLightInferenceCheckpointTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        set_hparams(config='configs/dslight/template.yaml', print_hparams=False)
        hparams['work_dir'] = str(self.root)
        hparams['exp_name'] = 'inference_checkpoint_test'
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

    def enable_lora(self, path):
        hparams['lora'] = {
            'enabled': True, 'base_ckpt': str(path), 'target_modules': ['linear'],
            'rank': 2, 'alpha': 2,
        }

    def infer_model(self):
        return DiffSingerAcousticInfer(device='cpu', load_vocoder=False).model

    def test_matching_dslight_lora_base_loads(self):
        self.enable_lora(self.write_checkpoint())
        self.assertEqual(self.infer_model().backbone_type, 'dslight')

    def test_legacy_lora_base_is_rejected_for_dslight(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        path = self.write_checkpoint()
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        self.enable_lora(path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

    def test_lora_rejects_wrong_adapter_shape(self):
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        model = DiffSingerAcoustic(vocab_size=len(load_phoneme_dictionary()), out_dims=128)
        inject_lora(model, rank=2, alpha=2, target_modules=['linear'])
        adapter = extract_lora_state_dict(model)
        adapter_key = next(key for key in adapter if key.endswith('.lora_A'))
        adapter[adapter_key] = adapter[adapter_key][:-1]
        torch.save({'category': 'acoustic', 'state_dict': adapter}, self.root / 'model_ckpt_steps_2.ckpt')
        self.enable_lora(base_path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

    def test_lora_adapter_only_update_loads(self):
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        model = DiffSingerAcoustic(vocab_size=len(load_phoneme_dictionary()), out_dims=128)
        inject_lora(model, rank=2, alpha=2, target_modules=['linear'])
        adapter = extract_lora_state_dict(model)
        torch.save({'category': 'acoustic', 'state_dict': adapter}, self.root / 'model_ckpt_steps_2.ckpt')
        self.enable_lora(base_path)
        self.assertEqual(self.infer_model().backbone_type, 'dslight')

    def test_lora_rejects_wrong_preferred_inference_checkpoint(self):
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        shutil.copy2(base_path, self.root / 'model_ckpt_steps_2.ckpt')
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        wrong_path = self.write_checkpoint()
        shutil.copy2(wrong_path, self.root / 'model_ckpt_steps_2.infer.ckpt')
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        self.enable_lora(base_path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

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
        self.enable_lora(base_path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

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
        self.enable_lora(base_path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

    def test_legacy_lora_rejects_dslight_base(self):
        path = self.write_checkpoint()
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        self.enable_lora(path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

    def test_lora_rejects_wrong_latest_full_checkpoint(self):
        base_path = self.root / 'base.ckpt'
        shutil.copy2(self.write_checkpoint(), base_path)
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        self.write_checkpoint()
        hparams.clear()
        hparams.update(copy.deepcopy(self.light_config))
        self.enable_lora(base_path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

    def test_wrong_dslight_dimensions_are_rejected(self):
        path = self.write_checkpoint()
        hparams['backbone_args']['num_channels'] = 128
        self.enable_lora(path)
        with self.assertRaises(RuntimeError):
            self.infer_model()

    def test_legacy_configuration_still_loads(self):
        hparams['backbone_type'] = self.legacy_config['backbone_type']
        hparams['backbone_args'] = self.legacy_config['backbone_args']
        hparams['shallow_diffusion_args']['aux_decoder_arch'] = 'convnext'
        hparams['shallow_diffusion_args']['aux_decoder_args'] = (
            self.legacy_config['shallow_diffusion_args']['aux_decoder_args']
        )
        self.enable_lora(self.write_checkpoint())
        self.assertEqual(self.infer_model().backbone_type, 'lynxnet2')


if __name__ == '__main__':
    unittest.main()
