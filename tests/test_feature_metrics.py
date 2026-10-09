import math
import unittest

import torch

from conditional_drifting.feature_metrics import (
    difference_weights, embedding_scores, empirical_embedding_gram,
    gaussian_embedding_gram, moment_scores, perturbed_inputs,
    embedding_matrix_scores, pathwise_embedding_matrices, polynomial_features,
    polynomial_embedding_gram, sample_input_jacobians,
)


class FeatureMetricTests(unittest.TestCase):
    def test_polynomial_kernel_and_normalization(self):
        y = torch.tensor([[1., 2.], [3., -.4]], dtype=torch.float64)
        phi = polynomial_features(y)
        inner = y @ y.T / 2
        torch.testing.assert_close(phi @ phi.T, inner + inner.square())

    def test_sample_input_jacobians(self):
        x = torch.tensor([.4, -.7], dtype=torch.float64)
        def sample(v):
            return torch.stack((v[:, 0].sin() + v[:, 1], v[:, 1].square()), dim=1) + torch.randn_like(v)
        y, jac = sample_input_jacobians(sample, x, 11, 17)
        expected = torch.tensor([[math.cos(.4), 1.], [0., -1.4]], dtype=x.dtype)
        torch.testing.assert_close(jac, expected.expand(11, -1, -1))
        torch.manual_seed(17)
        torch.testing.assert_close(y, sample(x[None].repeat(11, 1)))

    def test_pathwise_kernel_matches_finite_differences(self):
        torch.manual_seed(19)
        x = torch.tensor([.4, -.2], dtype=torch.float64)
        noise_p, noise_q = torch.randn(9, 2, dtype=x.dtype), torch.randn(13, 2, dtype=x.dtype)
        p = x + noise_p
        q = x.square() + .7 * noise_q
        jp = torch.eye(2, dtype=x.dtype).expand(9, -1, -1)
        jq = torch.diag(2 * x).expand(13, -1, -1)
        matrices = pathwise_embedding_matrices(p, q, jp, jq, [.5, 1.3])
        h = 1e-4
        queries = perturbed_inputs(x, h)
        # Unequal empirical sample counts also exercise signed normalization.
        clouds = [v[None] + noise_p for v in queries] + [v[None].square() + .7 * noise_q for v in queries]
        gram = torch.stack([torch.stack([
            sum(torch.exp(-torch.cdist(a, b).square() / (2 * l*l)).mean() for l in [.5, 1.3]) / 2
            for b in clouds]) for a in clouds])
        means = torch.stack([polynomial_features(c).mean(0) for c in clouds])
        moment_gram = means @ means.T
        for key, g in {"rbf": gram, "moments": moment_gram, "augmented": gram + moment_gram}.items():
            fd = difference_weights(2, h).T @ g @ difference_weights(2, h)
            torch.testing.assert_close(matrices[key][1:, 1:], fd[1:, 1:], atol=2e-7, rtol=1e-5)
            self.assertAlmostEqual(matrices[key][0, 0].item(), fd[0, 0].item(), places=10)

    def test_pathwise_identical_samples_zero(self):
        torch.manual_seed(13)
        y, jac = torch.randn(8, 2), torch.randn(8, 2, 3)
        for matrix in pathwise_embedding_matrices(y, y, jac, jac, [1.]).values():
            result = embedding_matrix_scores(matrix)
            self.assertLess(result["mmd"], 1e-7)
            self.assertLess(result["embedding_derivative_op"], 1e-7)

    def test_moment_value_detects_variance_growth(self):
        p = torch.tensor([[-1.], [1.]], dtype=torch.float64)
        values = []
        for scale in (1., 3., 10.):
            q = p * scale
            matrix = pathwise_embedding_matrices(p, q, torch.ones(2, 1, 1), torch.ones(2, 1, 1), [1.])
            values.append(embedding_matrix_scores(matrix["moments"])["mmd"])
        torch.testing.assert_close(torch.tensor(values), torch.tensor([0., 8., 99.]))

    def test_query_order_and_derivative_weights(self):
        x = torch.tensor([.4, -.2], dtype=torch.float64)
        queries = perturbed_inputs(x, .01)
        values = torch.cat((queries, 2 * queries))
        actual = difference_weights(2, .01).T @ values
        torch.testing.assert_close(actual[0], -x)
        torch.testing.assert_close(actual[1:], -torch.eye(2, dtype=x.dtype))

    def test_full_kernel_matches_direct_feature_sums(self):
        torch.manual_seed(9)
        cloud = torch.randn(6, 7, 1, dtype=torch.float64)
        gram = empirical_embedding_gram(cloud, [.4, 1.2])
        flat = cloud.flatten(0, 1)
        dist = torch.cdist(flat, flat).square()
        kernel = (torch.exp(-dist / (2 * .4**2)) + torch.exp(-dist / (2 * 1.2**2))) / 2
        weights = difference_weights(1, .2)
        expanded = weights.repeat_interleave(7, dim=0) / 7
        torch.testing.assert_close(weights.T @ gram @ weights, expanded.T @ kernel @ expanded)
        torch.testing.assert_close(gram, empirical_embedding_gram(cloud, [.4, 1.2], groups_per_block=6))

    def test_identical_cloud_zero(self):
        torch.manual_seed(12)
        p = torch.randn(5, 32, 2)
        result = embedding_scores(empirical_embedding_gram(torch.cat((p, p)), [1]), 2, .1)
        self.assertLess(result["mmd"], 1e-7)
        self.assertLess(result["embedding_derivative_op"], 1e-7)
        self.assertTrue(all(v == 0 for v in moment_scores(p, p, 2, .1).values()))

    def test_gaussian_formula_and_label_flip(self):
        means = torch.tensor([[.7], [-.7]], dtype=torch.float64)
        gram = gaussian_embedding_gram(means, .25, [1.])
        self.assertAlmostEqual(gram[0, 0].item(), (1 / 1.5)**.5)
        self.assertAlmostEqual(gram[0, 1].item(), (1 / 1.5)**.5 * math.exp(-1.4**2 / 3))
        x = torch.tensor([1.], dtype=torch.float64)
        queries = perturbed_inputs(x, .001)
        scores = embedding_scores(gaussian_embedding_gram(torch.cat((queries, -queries)), .25, [1.]), 1, .001)
        self.assertGreater(scores["mmd"], .5)
        self.assertGreater(scores["embedding_derivative_op"], .1)

    def test_fixed_power_oscillation(self):
        x = torch.tensor([math.sqrt(3) / 2, .5], dtype=torch.float64)
        amplitude, sigma, length = 2 / math.sqrt(3), .5, 1.
        exact = amplitude * math.sqrt((length**2 / (length**2 + 2 * sigma**2)) / (length**2 + 2 * sigma**2))
        for frequency in (4 * math.pi, 16 * math.pi):
            h = 1e-4
            p = perturbed_inputs(x, h)
            q = p.clone()
            q[:, 0] += amplitude / frequency * torch.sin(frequency * p[:, 1])
            scores = embedding_scores(gaussian_embedding_gram(torch.cat((p, q)), sigma**2, [length]), 2, h)
            self.assertLess(scores["mmd"], 1e-7)
            self.assertAlmostEqual(scores["embedding_derivative_op"], exact, places=4)

    def test_moment_matched_distributions_need_distribution_metric(self):
        # N(0,1) versus equiprobable atoms +/-1: equal first two moments.
        means = torch.tensor([[0.], [-1.], [1.]], dtype=torch.float64)
        gram = gaussian_embedding_gram(means, [1., 0., 0.], [1.])
        weights = torch.tensor([1., -.5, -.5], dtype=torch.float64)
        self.assertGreater((weights @ gram @ weights).item(), .01)

    def test_shared_noise_identity_has_zero_moment_derivative_error(self):
        torch.manual_seed(77)
        queries = perturbed_inputs(torch.tensor([.5, .7]), .1)
        p = queries[:, None] + torch.randn(1, 128, 2)
        q = queries[:, None] + torch.randn(1, 128, 2)
        scores = moment_scores(p, q, 2, .1)
        self.assertGreater(scores["mean_l2"], .01)
        self.assertLess(scores["mean_derivative_op"], 1e-6)
        self.assertLess(scores["covariance_derivative_op"], 1e-6)


if __name__ == "__main__":
    unittest.main()
