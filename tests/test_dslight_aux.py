import unittest

import torch
from torch import nn

from modules.aux_decoder import AuxDecoderAdaptor, build_aux_decoder, build_aux_loss


class DSLightAuxTest(unittest.TestCase):
    def test_parameter_budgets(self):
        for channels, layers, kernel, expected in (
            (96, 3, 5, 199424), (128, 4, 7, 415488), (192, 6, 7, 1118912),
        ):
            with self.subTest(channels=channels):
                model = build_aux_decoder(256, 128, 'dslight', {
                    'num_channels': channels, 'num_layers': layers, 'kernel_size': kernel,
                })
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_adaptor_and_denormalization(self):
        for features in (1, 2):
            with self.subTest(features=features):
                bounds = [0.0] * 5 if features == 1 else [[0.0] * 5] * features
                upper = [2.0] * 5 if features == 1 else [[2.0] * 5] * features
                model = AuxDecoderAdaptor(6, 5, features, bounds, upper, 'dslight', {
                    'num_channels': 8, 'num_layers': 2, 'dilation_cycle': [1, 3],
                }).eval()
                condition = torch.randn(2, 11, 6)
                raw = model(condition, infer=False)
                result = model(condition, infer=True)
                expected_shape = (2, 11, 5) if features == 1 else (2, 2, 11, 5)
                self.assertEqual(tuple(result.shape), expected_shape)
                torch.testing.assert_close(result, raw + 1)
        self.assertIsInstance(build_aux_loss('dslight'), nn.L1Loss)

    def test_unknown_arguments_rejected(self):
        with self.assertRaises(TypeError):
            build_aux_decoder(6, 5, 'dslight', {'num_layer': 2})


if __name__ == '__main__':
    unittest.main()
