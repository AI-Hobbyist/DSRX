import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from deployment.exporters.onnx_export import export_onnx
from modules.aux_decoder import build_aux_decoder
from modules.backbones import build_backbone


class DSLightOnnxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_opset15_dynamic_frames(self):
        torch.manual_seed(12)
        backbone = build_backbone(8, 2, 'dslight', {
            'num_channels': 16, 'num_layers': 2, 'cond_channels': 12,
            'time_embed_dim': 8, 'dilation_cycle': [1, 3],
        }).eval()
        decoder = build_aux_decoder(12, 8, 'dslight', {
            'num_channels': 16, 'num_layers': 2, 'dilation_cycle': [1, 3],
        }).eval()
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            root = Path(directory)
            spec = torch.randn(1, 2, 8, 11)
            step = torch.tensor([500.0])
            cond = torch.randn(1, 12, 11)
            export_onnx(
                backbone, (spec, step, cond), str(root / 'backbone.onnx'),
                input_names=['spec', 'step', 'condition'], output_names=['output'],
                dynamic_axes={'spec': {3: 'frames'}, 'condition': {2: 'frames'},
                              'output': {3: 'frames'}}, opset_version=15,
            )
            export_onnx(
                decoder, (cond.transpose(1, 2),), str(root / 'aux.onnx'),
                input_names=['condition'], output_names=['output'],
                dynamic_axes={'condition': {1: 'frames'}, 'output': {1: 'frames'}},
                opset_version=15,
            )
            for name in ('backbone', 'aux'):
                onnx.checker.check_model(onnx.load(root / f'{name}.onnx'))
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            sessions = {
                name: ort.InferenceSession(str(root / f'{name}.onnx'), sess_options=opts,
                                           providers=['CPUExecutionProvider'])
                for name in ('backbone', 'aux')
            }
            for frames in (7, 16):
                spec = torch.randn(1, 2, 8, frames)
                cond = torch.randn(1, 12, frames)
                actual = sessions['backbone'].run(None, {
                    'spec': spec.numpy(), 'step': step.numpy(), 'condition': cond.numpy(),
                })[0]
                np.testing.assert_allclose(actual, backbone(spec, step, cond).numpy(), rtol=1e-4, atol=1e-5)
                aux_cond = cond.transpose(1, 2).contiguous()
                actual = sessions['aux'].run(None, {'condition': aux_cond.numpy()})[0]
                np.testing.assert_allclose(actual, decoder(aux_cond).numpy(), rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
