import unittest

import torch

from conditional_drifting.sspa_solver_replay import barycenter, log_plan, residuals, variance_terms
from conditional_drifting.transport_reference import log_sinkhorn_cost


class ReplayTests(unittest.TestCase):
    def test_log_stabilizer_matches_primal(self):
        torch.manual_seed(71)
        kernel = torch.rand(3, 4, 5, dtype=torch.float64).clamp_min(.01)
        kernel[0, :, 3:] = 1e-8
        for steps in (1, 10, 30):
            u, v = torch.ones(3, 4, dtype=kernel.dtype), torch.ones(3, 5, dtype=kernel.dtype)
            for _ in range(steps):
                u = .25 / ((kernel @ v[:, :, None]).squeeze(-1) + 1e-8)
                v = .2 / ((kernel.transpose(1, 2) @ u[:, :, None]).squeeze(-1) + 1e-8)
            plan = u[:, :, None] * kernel * v[:, None, :]
            actual = log_plan(kernel.log(), steps, denominator=1e-8)["plan"]
            torch.testing.assert_close(actual, plan, rtol=1e-12, atol=1e-14)

    def test_reference_matches_scalar_and_cap_is_visible(self):
        torch.manual_seed(72)
        cost = torch.rand(3, 4, 5, dtype=torch.float64)
        result = log_plan(-cost / .3, 1000, tolerance=1e-10)
        self.assertTrue(result["converged"].all())
        for i in range(3):
            expected = log_sinkhorn_cost(cost[i], .3, tolerance=1e-10)["coupling"]
            torch.testing.assert_close(result["plan"][i], expected, atol=1e-10, rtol=1e-8)
        capped = log_plan(-cost / .001, 1, tolerance=1e-12)
        self.assertFalse(capped["converged"].all())
        self.assertTrue((capped["iterations"] == 1).all())

    def test_variance_identity(self):
        torch.manual_seed(73)
        source, velocity = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
        for scale in (0., .1, 1.):
            terms = variance_terms(source, velocity, scale)
            torch.testing.assert_close(terms["total"], terms["actual"], atol=1e-14, rtol=1e-12)
        terms = variance_terms(source, torch.ones_like(source))
        self.assertEqual(terms["second"].sum(), 0.)

    def test_independent_cloud_stopping(self):
        kernel = torch.tensor([[[0., 0.], [0., 0.]],
                               [[0., -6.], [-3., 0.]]], dtype=torch.float64)
        result = log_plan(kernel, 3000, tolerance=1e-10)
        self.assertTrue(result["converged"].all())
        self.assertEqual(result["iterations"][0], 10)
        self.assertGreater(result["iterations"][1], 10)
        for i in range(2):
            scalar = log_plan(kernel[i:i + 1], 3000, tolerance=1e-10)
            torch.testing.assert_close(result["plan"][i:i + 1], scalar["plan"])
            self.assertEqual(result["iterations"][i], scalar["iterations"][0])

    def test_opposite_fields_cancel_only_first_order_variance(self):
        source = torch.tensor([[[0., 0.], [1., 1.]]])
        velocity = torch.tensor([[[.1, .2], [-.2, .3]]])
        positive = variance_terms(source, velocity)
        negative = variance_terms(source, -velocity)
        torch.testing.assert_close(positive["first"], -negative["first"])
        torch.testing.assert_close(positive["second"], negative["second"])
        self.assertGreater(((positive["actual"] + negative["actual"]) / 2).item(), 0.)

    def test_readouts_separate_mass_error(self):
        plan = torch.tensor([[[.1, .1], [.3, .5]]], dtype=torch.float64)
        target = torch.tensor([[[1.], [2.]]], dtype=torch.float64)
        self.assertFalse(torch.allclose(barycenter(plan, target, "row"), barycenter(plan, target, "uniform")))
        torch.testing.assert_close(barycenter(plan, target, "production"), barycenter(plan, target, "row"), atol=1e-6, rtol=1e-6)
        self.assertGreater(residuals(plan).max(), .1)
