import math
import unittest

import torch

from conditional_drifting.losses import _batched_sinkhorn_barycentric_projection, _sinkhorn_barycentric_projection
from conditional_drifting.transport_reference import log_sinkhorn, log_sinkhorn_cost, marginal_residuals, sinkhorn_divergence
from scripts.run_transport_reference_audit import equilibrium_checks, finite_parameter_gradients


class TransportReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(319)

    def test_two_point_closed_form(self):
        x = torch.tensor([[-1.], [1.]], dtype=torch.float64)
        epsilon = .4
        result = log_sinkhorn(x, x, epsilon)
        off = .5/(1+math.exp(2/epsilon))
        expected = torch.tensor([[.5-off, off], [off, .5-off]], dtype=torch.float64)
        torch.testing.assert_close(result["coupling"], expected, atol=1e-12, rtol=1e-10)
        value = 1-epsilon*math.log(math.cosh(1/epsilon))
        self.assertAlmostEqual(result["objective"], value, places=12)

    def test_unequal_mass_constraints(self):
        x, y = torch.randn(3, 2), torch.randn(5, 2)
        result = log_sinkhorn(x, y, .6)
        self.assertTrue(result["converged"])
        self.assertLessEqual(max(marginal_residuals(result["coupling"]).values()), 1e-8)
        torch.testing.assert_close(result["barycenter"], 3*result["coupling"]@y.double())

    def test_explicit_failure_and_validation(self):
        x = torch.tensor([[-2.], [-2.], [-2.], [2.]])
        y = -x
        result = log_sinkhorn(x, y, .001, max_iterations=1, require_convergence=False)
        self.assertFalse(result["converged"])
        with self.assertRaises(RuntimeError):
            log_sinkhorn(x, y, .001, max_iterations=1)
        for epsilon in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                log_sinkhorn(x, y, epsilon)

    def test_row_shifts_preserve_reference(self):
        cost = torch.rand(4, 7, dtype=torch.float64)
        a = log_sinkhorn_cost(cost, .2)["coupling"]
        b = log_sinkhorn_cost(cost+torch.arange(4)[:, None], .2)["coupling"]
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)

    def test_production_diagnostics_do_not_change_default(self):
        x, y = torch.randn(4, 2), torch.randn(7, 2)
        kwargs = {"epsilon": .4, "min_epsilon": .001, "iterations": 30}
        a = _sinkhorn_barycentric_projection(x, y, y, **kwargs)
        b, diag = _sinkhorn_barycentric_projection(x, y, y, **kwargs, return_diagnostics=True)
        self.assertTrue(torch.equal(a, b))
        batch_default = _batched_sinkhorn_barycentric_projection(x[None], y[None], y[None], **kwargs)
        c, batched_diag = _batched_sinkhorn_barycentric_projection(x[None], y[None], y[None], **kwargs, return_diagnostics=True)
        self.assertTrue(torch.equal(batch_default, c))
        torch.testing.assert_close(b, c[0])
        torch.testing.assert_close(diag["coupling"], batched_diag["coupling"][0])

    def test_row_normalization_does_not_fix_target_mass(self):
        x = torch.tensor([[-2.], [-2.], [-2.], [2.]])
        y = -x
        _, diag = _sinkhorn_barycentric_projection(x, y, y, epsilon=.001, min_epsilon=.001,
                                                   iterations=1, return_diagnostics=True)
        residual = marginal_residuals(diag["row_weights"]/4)
        self.assertLess(residual["row_relative"], 1e-6)
        self.assertGreater(residual["column_relative"], .1)
        self.assertTrue(diag["kernel_floor_mask"].any())

    def test_collapsed_equilibrium(self):
        self.assertTrue(all(r["passed"] for r in equilibrium_checks()))

    def test_parameter_gradient_and_mse_scaling(self):
        result = finite_parameter_gradients()
        self.assertTrue(result["passed"])
        self.assertLess(result["finite_difference_max_absolute_error"], 1e-7)
        self.assertLess(result["mse_identity_max_absolute_error"], 1e-8)
        self.assertGreater(result["independent_reference_drift_difference"], 1e-4)

    def test_divergence_zero_for_same_cloud(self):
        x = torch.randn(5, 3, dtype=torch.float64)
        self.assertAlmostEqual(sinkhorn_divergence(x, x, .5), 0., places=12)


if __name__ == "__main__":
    unittest.main()
