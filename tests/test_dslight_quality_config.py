import unittest

from modules.aux_decoder import build_aux_decoder
from modules.backbones import build_backbone
from utils.hparams import set_hparams


class DSLightQualityConfigTest(unittest.TestCase):
    def test_quality_configuration_builds_expected_architectures(self):
        config = set_hparams(config='configs/templates/config_acoustic_dslight_quality.yaml', print_hparams=False)
        shallow = config['shallow_diffusion_args']
        backbone = build_backbone(128, 1, config['backbone_type'], config['backbone_args'])
        decoder = build_aux_decoder(256, 128, shallow['aux_decoder_arch'], shallow['aux_decoder_args'])
        self.assertEqual(sum(parameter.numel() for parameter in backbone.parameters()), 5177472)
        self.assertEqual(sum(parameter.numel() for parameter in decoder.parameters()), 1118912)
        self.assertEqual(shallow['aux_decoder_grad'], 1.0)
        self.assertEqual(config['optimizer_args']['optimizer_cls'], 'torch.optim.AdamW')
        self.assertEqual(config['optimizer_args']['lr'], 0.0006)
        self.assertEqual(config['optimizer_args']['weight_decay'], 0.01)
        self.assertEqual(set(config['optimizer_args']), {'optimizer_cls', 'lr', 'weight_decay'})
        self.assertEqual(config['max_batch_frames'], 16000)
        self.assertEqual(config['max_batch_size'], 24)
        self.assertGreater(config['lr_scheduler_args']['step_size'], 6000)


if __name__ == '__main__':
    unittest.main()
