import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_dslight_width import initialize_narrow
from modules.backbones.dslight import DSLightBackbone


class DSLightWidthTest(unittest.TestCase):
    def test_gate_only_reduction_preserves_residual_dimensions(self):
        teacher = DSLightBackbone(
            5, 1, num_channels=8, num_layers=2, cond_channels=6, time_embed_dim=8
        ).eval()
        student = DSLightBackbone(
            5,
            1,
            num_channels=8,
            num_layers=2,
            cond_channels=6,
            time_embed_dim=8,
            expansion_channels=4,
        ).eval()
        initialize_narrow(teacher, student)
        self.assertEqual(student.blocks[0].expand.out_channels, 8)
        self.assertEqual(student.blocks[0].project.in_channels, 8)
        torch.testing.assert_close(
            student.input_projection.weight, teacher.input_projection.weight
        )
        torch.testing.assert_close(
            student.output_projection.weight, teacher.output_projection.weight
        )
        self.assertLess(
            sum(x.numel() for x in student.parameters()),
            sum(x.numel() for x in teacher.parameters()),
        )

    def test_pruning_preserves_gate_halves_and_teacher(self):
        teacher = DSLightBackbone(
            5, 1, num_channels=8, num_layers=2, cond_channels=6, time_embed_dim=8
        ).eval()
        student = DSLightBackbone(
            5, 1, num_channels=4, num_layers=2, cond_channels=6, time_embed_dim=8
        ).eval()
        with torch.no_grad():
            teacher.output_projection.weight.copy_(
                torch.arange(8)
                .view(1, 8, 1)
                .expand_as(teacher.output_projection.weight)
            )
        original = {key: value.clone() for key, value in teacher.state_dict().items()}
        initialize_narrow(teacher, student)
        keep = torch.tensor([4, 5, 6, 7])
        gates = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
        torch.testing.assert_close(
            student.blocks[0].expand.weight,
            teacher.blocks[0].expand.weight[gates][:, keep],
        )
        torch.testing.assert_close(
            student.blocks[0].project.weight,
            teacher.blocks[0].project.weight[keep][:, gates],
        )
        torch.testing.assert_close(
            student.blocks[0].depthwise.weight, teacher.blocks[0].depthwise.weight[keep]
        )
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        with torch.no_grad():
            for frames in (7, 11):
                output = student(
                    torch.randn(1, 1, 5, frames),
                    torch.tensor([400.0]),
                    torch.randn(1, 6, frames),
                )
                self.assertEqual(output.shape, (1, 1, 5, frames))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
