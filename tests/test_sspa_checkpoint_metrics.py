import unittest

import torch

from conditional_drifting.sspa_checkpoint_metrics import moment_derivatives, output_jacobians


class MomentDerivativeTests(unittest.TestCase):
    def test_linear_mean_constant_covariance(self):
        torch.manual_seed(10)
        x = torch.randn(3, 2, dtype=torch.float64)
        z = torch.randn(3, 12, 2, dtype=torch.float64)
        matrix = torch.tensor([[2., 1.], [-1., 3.]], dtype=torch.float64)
        inputs = x.repeat_interleave(12, 0).requires_grad_(True)
        outputs = inputs @ matrix.T + z.flatten(0, 1)
        jac = output_jacobians(outputs, inputs).reshape(3, 12, 2, 2)
        _, _, jm, jc = moment_derivatives(outputs.reshape(3, 12, 2), jac)
        torch.testing.assert_close(jm, matrix.expand(3, 2, 2))
        torch.testing.assert_close(jc, torch.zeros_like(jc))

    def test_covariance_derivative_matches_finite_difference(self):
        torch.manual_seed(11)
        x = torch.randn(2, 2, dtype=torch.float64)
        z = torch.randn(2, 30, 2, dtype=torch.float64)
        def cloud(value):
            return value[:, None].square() + value[:, None].exp() * z
        inputs = x.repeat_interleave(30, 0).requires_grad_(True)
        outputs = inputs.square() + inputs.exp() * z.flatten(0, 1)
        jac = output_jacobians(outputs, inputs).reshape(2, 30, 2, 2)
        _, _, jm, jc = moment_derivatives(outputs.reshape(2, 30, 2), jac)
        for k in range(2):
            delta = torch.zeros_like(x)
            delta[:, k] = 1e-5
            plus, minus = cloud(x + delta), cloud(x - delta)
            def covariance(y):
                centered = y - y.mean(1, keepdim=True)
                return centered.transpose(1, 2) @ centered / 29
            torch.testing.assert_close(jm[:, :, k], (plus.mean(1) - minus.mean(1)) / 2e-5)
            torch.testing.assert_close(jc[:, :, :, k], (covariance(plus) - covariance(minus)) / 2e-5)

    def test_single_sample_rejected(self):
        with self.assertRaises(ValueError):
            moment_derivatives(torch.ones(2, 1, 3), torch.ones(2, 1, 3, 3))
