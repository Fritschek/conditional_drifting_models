import unittest

import torch

from scripts.run_learned_metric_gate import (
    lifted_probes, moment_vectors, refinement_decisions, sspa_reference, stream_seed,
)
from scripts.run_channel_metric_controls import expected_output_loss_gradient, sampled_loss_gradient, split_metrics


class LearnedMetricGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(51)
        self.x = torch.randn(3, 8, dtype=torch.float64)
        self.probes = lifted_probes(self.x)
        self.bw = [.325, .65, 1.3]

    def test_basis_and_dimensions(self):
        a, b = self.probes[0]["w"], self.probes[4]["w"]
        torch.testing.assert_close(a.norm(), a.new_tensor(1.))
        torch.testing.assert_close(a.dot(b), a.new_tensor(0.), atol=1e-14, rtol=0)
        self.assertEqual(len(self.probes), 8)
        with self.assertRaises(ValueError):
            lifted_probes(torch.ones(3, 8, dtype=torch.float64))

    def test_disjoint_streams(self):
        seeds = [stream_seed(n, a, r, role, model, s) for n in (512, 2048)
                 for a in range(3) for r in range(8) for role in range(3)
                 for model in range(4) for s in range(2)]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_gaussian_reference_quadratic_and_cosine(self):
        mean, jac, gradients = sspa_reference(self.x[0], .325, self.bw, self.probes)
        for probe in self.probes:
            w, offset = probe["w"], probe["offset"]
            if probe["kind"] == "quadratic":
                exact = jac.T@w*(mean.dot(w)-offset)
            elif probe["kind"] == "cosine":
                exact = jac.T@w*torch.sin(mean.dot(w)-offset)*torch.exp(w.new_tensor(-.325**2/4))
            else:
                continue
            torch.testing.assert_close(gradients[probe["name"]], exact)

    def test_probe_sample_gradients_by_autograd(self):
        y = torch.randn(19, 8, dtype=torch.float64, requires_grad=True)
        jac = torch.eye(8, dtype=torch.float64).expand(len(y), -1, -1)
        for p in self.probes:
            t = y@p["w"]-p["offset"]
            if p["kind"] == "quadratic":
                values = .5*t.square()
            elif p["kind"] == "cosine":
                values = 1-t.cos()
            elif p["kind"] == "logistic":
                values = torch.nn.functional.softplus(t)
            else:
                values = sum(torch.exp(-(y-p["center"]).square().sum(-1)/(2*l*l)) for l in self.bw)/len(self.bw)
            expected = torch.autograd.grad(values.mean(), y)[0].sum(0)
            torch.testing.assert_close(sampled_loss_gradient(y, jac, p, self.bw), expected)

    def test_integrated_rbf_reference_by_finite_difference(self):
        x = self.x[0]
        mean, jac, gradients = sspa_reference(x, .325, self.bw, self.probes)
        from conditional_drifting.channels import sspa
        for p in (self.probes[0], self.probes[4]):
            def value(u):
                m = sspa(u[None], 0., u.device)[0]
                return sum((l*l/(l*l+.325**2/2))**4
                           * torch.exp(-(m-p["center"]).square().sum()/(2*(l*l+.325**2/2)))
                           for l in self.bw)/len(self.bw)
            h = 1e-5
            fd = torch.stack([(value(x+h*d)-value(x-h*d))/(2*h) for d in torch.eye(8, dtype=x.dtype)])
            torch.testing.assert_close(fd, gradients[p["name"]], rtol=1e-6, atol=1e-10)

    def test_moment_derivatives(self):
        z = torch.randn(23, 8, dtype=torch.float64)
        matrices = torch.randn(23, 8, 8, dtype=torch.float64)
        directions = torch.stack((self.probes[0]["w"], self.probes[4]["w"]))
        def values(x):
            y = z+torch.einsum("noa,a->no", matrices, x)
            centered = y-y.mean(0)
            return centered.T@centered/(len(y)-1), (y@directions.T).pow(4).mean(0)
        x = self.x[0]
        actual = moment_vectors(z+torch.einsum("noa,a->no", matrices, x), matrices, directions)
        c, f = torch.autograd.functional.jacobian(values, x)
        torch.testing.assert_close(actual["covariance_derivative"], c)
        torch.testing.assert_close(actual["fourth_derivative"], f)

    def test_split_identity_with_custom_bandwidths(self):
        p, q = torch.randn(2, 12, 8, dtype=torch.float64)
        jac = torch.eye(8, dtype=torch.float64).expand(12, -1, -1)
        row = split_metrics((p, q, jac, jac), (q, p, jac, jac), 8, self.bw, detailed=True)
        self.assertLess(row["rbf_cross_trace"], 0)
        a = torch.tensor(row["rbf_self_matrix"])
        b = torch.tensor(row["rbf_cross_matrix"])
        torch.testing.assert_close(torch.tensor(row["rbf_noise_matrix"]), a-b)

    def test_refinement_rule(self):
        def row(anchor, order, agrees, cheap):
            return {"anchor": anchor, "target": {"ordering": order},
                    "kernel_agrees": agrees, "cheap_agrees": cheap}
        decisions = refinement_decisions([row(0, 0, False, []), row(1, 1, False, []), row(2, 1, True, [])])
        self.assertEqual([r["refine"] for r in decisions], [True, False, True])


if __name__ == "__main__":
    unittest.main()
