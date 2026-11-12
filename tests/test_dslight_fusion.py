import unittest

import torch
from torch.nn import functional as F

from modules.backbones.dslight_fusion import gated_projection


class DSLightFusionTest(unittest.TestCase):
    def test_projection_expression_and_gradients(self):
        for width in (4, 8):
            a = torch.randn(2, width, 11, requires_grad=True)
            b = torch.randn_like(a, requires_grad=True)
            weight = torch.randn(8, 2 * width, 1, requires_grad=True)
            bias = torch.randn(8, requires_grad=True)
            arguments = (a, b, weight, bias)
            expected = F.conv1d(torch.cat((a * b.sigmoid(), a), dim=1), weight, bias)
            actual = gated_projection(*arguments)
            torch.testing.assert_close(actual, expected)
            expected_grad = torch.autograd.grad(expected.square().mean(), arguments)
            actual_grad = torch.autograd.grad(actual.square().mean(), arguments)
            for left, right in zip(expected_grad, actual_grad):
                torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
