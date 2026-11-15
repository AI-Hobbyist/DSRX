import copy
import unittest

import torch

from modules.backbones.dslight import DSLightBackbone
from modules.backbones.dslight_quantization import (
    WeightOnlyInt8Pointwise,
    load_weight_only_package,
    quantize_dslight_pointwise,
)


class DSLightQuantizationTest(unittest.TestCase):
    def test_manifest_roundtrip_and_protected_layers(self):
        model = DSLightBackbone(
            5, 1, num_channels=8, num_layers=2, cond_channels=6, time_embed_dim=8
        ).eval()
        restored = copy.deepcopy(model)
        layers = quantize_dslight_pointwise(model)
        self.assertEqual(len(layers), 6)
        self.assertIsInstance(model.blocks[0].depthwise, torch.nn.Conv1d)
        self.assertIsInstance(model.output_projection, torch.nn.Conv1d)
        self.assertIsInstance(model.blocks[0].project, WeightOnlyInt8Pointwise)
        package = {
            "format": "dslight-int8-weight-only-v1",
            "layers": layers,
            "state_dict": model.state_dict(),
        }
        load_weight_only_package(restored, package)
        with torch.inference_mode():
            for frames in (7, 11):
                x = torch.randn(1, 1, 5, frames)
                t = torch.tensor([400.0])
                cond = torch.randn(1, 6, frames)
                expected = model(x, t, cond)
                torch.testing.assert_close(
                    expected, restored(x, t, cond), rtol=0, atol=0
                )
                prepared = model.prepare_condition(cond)
                torch.testing.assert_close(expected, model(x, t, prepared))
                traced = torch.jit.trace(model, (x, t, cond))
                torch.testing.assert_close(expected, traced(x, t, cond))

    def test_zero_rows_and_training_rejection(self):
        source = torch.nn.Conv1d(3, 4, 1)
        source.weight.data.zero_()
        quantized = WeightOnlyInt8Pointwise(source)
        x = torch.randn(1, 3, 7)
        torch.testing.assert_close(source(x), quantized(x))
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            quantized.train()(x)


if __name__ == "__main__":
    unittest.main()
