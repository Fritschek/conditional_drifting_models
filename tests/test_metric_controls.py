import math
import unittest

import torch

from conditional_drifting.feature_metrics import (
    difference_weights, gaussian_embedding_gram, pathwise_cross_embedding_matrices,
    pathwise_embedding_matrices, perturbed_inputs,
)
from scripts.run_channel_metric_controls import (
    ANCHORS, BANDWIDTHS, SIGMA, expected_output_loss_gradient, gaussian_population_matrices,
    gradient_comparison, perturbed_mean, probes, rotation_channel, rotation_samples,
)


class MetricControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(913)

    def clouds(self):
        return (torch.randn(3, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64),
                torch.randn(3, 2, 2, dtype=torch.float64), torch.randn(5, 2, 2, dtype=torch.float64))

    def test_tiled_self_matches_original(self):
        clouds = self.clouds()
        expected = pathwise_embedding_matrices(*clouds, BANDWIDTHS)
        for block in (1, 4, 20):
            actual = pathwise_cross_embedding_matrices(clouds, clouds, BANDWIDTHS, block)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])

    def test_cross_transpose_and_polarization(self):
        a, b = self.clouds(), self.clouds()
        ab = pathwise_cross_embedding_matrices(a, b, BANDWIDTHS, 2)
        ba = pathwise_cross_embedding_matrices(b, a, BANDWIDTHS, 2)
        aa, bb = pathwise_embedding_matrices(*a, BANDWIDTHS), pathwise_embedding_matrices(*b, BANDWIDTHS)
        pool = pathwise_embedding_matrices(*[torch.cat((x, y)) for x, y in zip(a, b)], BANDWIDTHS)
        for key in ab:
            torch.testing.assert_close(ab[key], ba[key].T)
            torch.testing.assert_close((ab[key]+ab[key].T)/2, 2*pool[key]-(aa[key]+bb[key])/2)

    def test_negative_cross_trace_not_clipped(self):
        p, q, jp, jq = self.clouds()
        result = pathwise_cross_embedding_matrices((p, q, jp, jq), (q, p, jq, jp), BANDWIDTHS)
        for matrix in result.values():
            self.assertLess(matrix[1:, 1:].trace().item(), 0)

    def test_rotation_jacobian_and_finite_difference(self):
        anchor = torch.tensor([.6, -.8], dtype=torch.float64)
        z = torch.randn(5, 2, dtype=torch.float64)
        for omega in (0., 1., 4., 16.):
            x = anchor.repeat(5, 1).requires_grad_(True)
            y = rotation_channel(x, z, omega, anchor)
            actual = torch.stack([torch.autograd.grad(y[:, j].sum(), x, retain_graph=True)[0] for j in range(2)], 1)
            expected_y, expected_jac = rotation_samples(anchor, z, omega)
            torch.testing.assert_close(y, expected_y)
            torch.testing.assert_close(actual, expected_jac)
            h = 1e-5
            for j in range(2):
                step = torch.eye(2, dtype=x.dtype)[j]*h
                fd = (rotation_channel(x+step, z, omega, anchor)-rotation_channel(x-step, z, omega, anchor))/(2*h)
                torch.testing.assert_close(fd, actual[:, :, j], atol=1e-7, rtol=1e-7)

    def test_rotation_preserves_noise_norm_away_from_anchor(self):
        anchor = torch.tensor([.6, -.8], dtype=torch.float64)
        x, z = torch.randn(50, 2, dtype=torch.float64), torch.randn(50, 2, dtype=torch.float64)
        rotated = (rotation_channel(x, z, 16., anchor)-x)/SIGMA
        torch.testing.assert_close(rotated.square().sum(1), z.square().sum(1))

    def test_linear_variance_formula(self):
        n, reps, omega = 256, 1024, 4.
        mean = torch.randn(reps, n, 2, dtype=torch.float64).mean(1)
        measured = (SIGMA*omega)**2*mean.square().sum(1).mean().item()
        expected = 2*SIGMA**2*omega**2/n
        self.assertLess(abs(measured/expected-1), .1)

    def test_bump_values_and_jacobians_at_all_anchors(self):
        anchors = torch.tensor(ANCHORS, dtype=torch.float64)
        u, v = torch.tensor([2**-.5, 2**-.5]), torch.tensor([2**-.5, -2**-.5])
        u, v = u.double(), v.double()
        for slope in (-4., 0., 4.):
            for anchor in anchors:
                f = lambda x: perturbed_mean(x, anchors, .1, slope, u, v)
                torch.testing.assert_close(f(anchor), anchor+.1*u)
                torch.testing.assert_close(torch.autograd.functional.jacobian(f, anchor), torch.eye(2, dtype=anchor.dtype)+slope*u[:, None]*v[None])

    def test_gaussian_population_derivative_against_finite_difference(self):
        x = torch.tensor([.6, -.8], dtype=torch.float64)
        jp = torch.eye(2, dtype=x.dtype)
        jq = torch.tensor([[2., .3], [-.2, 1.]], dtype=x.dtype)
        q = x+torch.tensor([.1, 0.], dtype=x.dtype)
        exact = gaussian_population_matrices(x, q, jp, jq)["rbf"]
        h = 1e-4
        pqueries = perturbed_inputs(x, h)
        qqueries = q+(pqueries-x)@jq.T
        gram = gaussian_embedding_gram(torch.cat((pqueries, qqueries)), SIGMA**2, BANDWIDTHS)
        w = difference_weights(2, h)
        finite = w.T@gram@w
        torch.testing.assert_close(exact[1:, 1:], finite[1:, 1:], rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(exact[0, 0].item(), finite[0, 0].item(), places=12)

    def test_population_value_fixed_while_derivative_changes(self):
        x = torch.tensor([1., 0.], dtype=torch.float64)
        eye = torch.eye(2, dtype=x.dtype)
        values = []
        for slope in (-4., 0., 4.):
            jq = eye.clone()
            jq[0, 0] += slope
            matrix = gaussian_population_matrices(x, x, eye, jq)["rbf"]
            self.assertLess(abs(matrix[0, 0].item()), 1e-12)
            values.append(matrix[1:, 1:].trace().item())
        self.assertAlmostEqual(values[0], values[2])
        self.assertGreater(values[0], 1.)
        self.assertLess(abs(values[1]), 1e-12)

    def test_task_identity_and_quadrature(self):
        mean = torch.tensor([.4, -.8], dtype=torch.float64)
        for probe in probes("cpu"):
            g = expected_output_loss_gradient(mean, probe)
            stats = gradient_comparison(g, .2*g)
            self.assertAlmostEqual(stats["relative_gradient_error"]**2,
                                   1+stats["norm_ratio"]**2-2*stats["norm_ratio"]*stats["cosine"])
            if probe["kind"] == "logistic":
                torch.testing.assert_close(g, expected_output_loss_gradient(mean, probe, 64), atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
