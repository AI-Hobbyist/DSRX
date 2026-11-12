import unittest

import torch

from modules.backbones import build_backbone, run_backbone


class DSLightBackboneTest(unittest.TestCase):
    def test_reflow_prepares_once_per_sampling_invocation(self):
        from modules.core.reflow import RectifiedFlow
        from utils.hparams import hparams

        previous = dict(hparams)
        try:
            hparams.update(sampling_algorithm='rk2', sampling_steps=3, infer=False)
            model = RectifiedFlow(
                5, backbone_type='dslight',
                backbone_args={'num_channels': 8, 'num_layers': 2,
                               'cond_channels': 6, 'time_embed_dim': 8},
                spec_min=[-1.] * 5, spec_max=[1.] * 5,
            ).eval()
            calls = []
            hook = model.velocity_fn.condition_projection.register_forward_hook(
                lambda *args: calls.append(1)
            )
            try:
                for frames in (11, 7):
                    model.inference(torch.randn(1, 6, frames), device='cpu')
            finally:
                hook.remove()
            self.assertEqual(len(calls), 2)
        finally:
            hparams.clear()
            hparams.update(previous)

    def test_prepared_condition_matches_forward_and_is_utterance_local(self):
        model = build_backbone(5, 1, 'dslight', {
            'num_channels': 8, 'num_layers': 2, 'cond_channels': 6,
            'time_embed_dim': 8,
        }).eval()
        state_keys = tuple(model.state_dict())
        with torch.no_grad():
            for frames in (11, 7, 11):
                spec = torch.randn(2, 1, 5, frames)
                cond = torch.randn(2, 6, frames)
                prepared = model.prepare_condition(cond)
                for step in (torch.tensor([1, 2]), torch.tensor([10, 20])):
                    torch.testing.assert_close(
                        model(spec, step, prepared), model(spec, step, cond)
                    )
        self.assertEqual(state_keys, tuple(model.state_dict()))

    def test_parameter_budgets(self):
        for channels, layers, expected in ((128, 8, 759424), (192, 12, 2389376), (256, 15, 5177472)):
            with self.subTest(channels=channels):
                model = build_backbone(128, 1, 'dslight', {
                    'num_channels': channels, 'num_layers': layers,
                })
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_forward_dispatch_and_trace(self):
        model = build_backbone(5, 2, 'dslight', {
            'num_channels': 8, 'num_layers': 2, 'cond_channels': 6,
            'time_embed_dim': 8, 'dilation_cycle': [1, 3],
        }).eval()
        spec = torch.randn(2, 2, 5, 11)
        step = torch.tensor([1, 2])
        cond = torch.randn(2, 6, 11)
        with torch.no_grad():
            output = run_backbone(model, spec, step, cond, valid_mask=torch.ones(2, 11))
            traced = torch.jit.trace(model, (spec, step, cond))
            self.assertEqual(output.shape, spec.shape)
            torch.testing.assert_close(output, traced(spec, step, cond))
        self.assertFalse(model.supports_valid_mask)

    def test_unknown_arguments_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown DSLight'):
            build_backbone(5, 1, 'dslight', {'num_layer': 2})


if __name__ == '__main__':
    unittest.main()
