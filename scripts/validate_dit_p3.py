import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.modules import dit as deployment_dit
from deployment.exporters.onnx_export import export_onnx
from deployment.modules.rectified_flow import RectifiedFlowONNX
from modules.backbones.dit import DiT
from utils.hparams import hparams


def build_dit() -> DiT:
    hparams.clear()
    hparams.update({'hidden_size': 8})
    model = DiT(
        4,
        1,
        num_layers=2,
        num_channels=8,
        num_heads=2,
        mlp_ratio=2,
        time_embed_dim=8,
        use_gradient_checkpointing=False,
    ).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[1].bias.normal_()
        model.final_modulation[1].bias.normal_()
        model.output_proj.weight.normal_()
    return model


def validate_adapter_contract() -> None:
    model = build_dit()
    adapter = deployment_dit.prepare_backbone_for_onnx(model)
    assert isinstance(adapter, deployment_dit.DiTONNXAdapter)

    for frames in (1, 7, 16):
        spec = torch.randn(1, 1, 4, frames)
        step = torch.tensor([500.0])
        cond = torch.randn(1, 8, frames)
        expected = model(
            spec,
            step,
            cond,
            valid_mask=torch.ones(1, frames, dtype=torch.bool),
        )
        actual = adapter(spec, step, cond)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    class LegacyBackbone(nn.Module):
        def forward(self, spec, diffusion_step, cond):
            return spec + diffusion_step[:, None, None, None] * 0 + cond[:, None, :1, :] * 0

    legacy = LegacyBackbone()
    assert deployment_dit.prepare_backbone_for_onnx(legacy) is legacy
    legacy_inputs = (
        torch.randn(1, 1, 4, 7),
        torch.tensor([500.0]),
        torch.randn(1, 8, 7),
    )
    compiled_legacy = deployment_dit.compile_backbone_for_onnx(
        legacy, legacy_inputs
    )
    torch.testing.assert_close(compiled_legacy(*legacy_inputs), legacy(*legacy_inputs))


def validate_dynamic_onnx(runtime: bool) -> None:
    backbone = build_dit()
    model = deployment_dit.prepare_backbone_for_onnx(backbone)
    spec = torch.randn(1, 1, 4, 7)
    step = torch.tensor([500.0])
    cond = torch.randn(1, 8, 7)
    compiled = deployment_dit.compile_backbone_for_onnx(
        backbone,
        (spec, step, cond),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / 'dit.onnx'
        export_onnx(
            compiled,
            (spec, step, cond),
            model_path,
            input_names=['spec', 'diffusion_step', 'condition'],
            output_names=['prediction'],
            dynamic_axes={
                'spec': {3: 'n_frames'},
                'condition': {2: 'n_frames'},
                'prediction': {3: 'n_frames'},
            },
            opset_version=15,
        )
        graph = onnx.load(model_path)
        onnx.checker.check_model(graph)
        operator_types = {node.op_type for node in graph.graph.node}
        assert 'ScaledDotProductAttention' not in operator_types
        assert 'MatMul' in operator_types and 'Softmax' in operator_types

        if runtime:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider'],
            )
            for frames in (7, 1, 16):
                inputs = {
                    'spec': np.random.randn(1, 1, 4, frames).astype(np.float32),
                    'diffusion_step': np.array([500.0], dtype=np.float32),
                    'condition': np.random.randn(1, 8, frames).astype(np.float32),
                }
                expected = model(
                    torch.from_numpy(inputs['spec']),
                    torch.from_numpy(inputs['diffusion_step']),
                    torch.from_numpy(inputs['condition']),
                ).detach().numpy()
                scripted = compiled(
                    torch.from_numpy(inputs['spec']),
                    torch.from_numpy(inputs['diffusion_step']),
                    torch.from_numpy(inputs['condition']),
                ).detach().numpy()
                np.testing.assert_allclose(
                    scripted,
                    expected,
                    atol=1e-6,
                    rtol=1e-5,
                    err_msg=f'TorchScript frames={frames}',
                )
                actual = session.run(None, inputs)[0]
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=2e-5,
                    rtol=2e-4,
                    err_msg=f'frames={frames}',
                )


def validate_sampler_wrapper() -> None:
    flow = RectifiedFlowONNX(
        out_dims=4,
        num_feats=1,
        t_start=0.0,
        time_scale_factor=1000,
        backbone_type='dit',
        backbone_args={
            'num_layers': 2,
            'num_channels': 8,
            'num_heads': 2,
            'mlp_ratio': 2,
            'time_embed_dim': 8,
            'patch_size': 1,
            'rope_base': 10000.0,
            'layer_norm_eps': 1e-6,
            'attention_dropout': 0.0,
            'mlp_dropout': 0.0,
            'use_gradient_checkpointing': False,
        },
        spec_min=[-1.0] * 4,
        spec_max=[1.0] * 4,
    ).eval()
    noise = torch.randn(1, 1, 4, 7)
    step = torch.tensor([500.0])
    condition = torch.randn(1, 8, 7)
    flow.set_backbone(
        deployment_dit.compile_backbone_for_onnx(
            flow.backbone,
            (noise, step, condition),
        )
    )
    scripted = torch.jit.script(flow)
    for frames in (1, 7, 16):
        output = scripted(
            torch.randn(1, frames, 8),
            torch.randn(1, frames, 4),
            torch.tensor(0.4),
            2,
        )
        assert output.shape == (1, frames, 4)
        assert torch.isfinite(output).all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(1234)
    validate_adapter_contract()
    validate_dynamic_onnx(args.runtime)
    validate_sampler_wrapper()
    print('DiT P3 validation passed.')


if __name__ == '__main__':
    main()