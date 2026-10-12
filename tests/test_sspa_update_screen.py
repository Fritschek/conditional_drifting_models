import copy
from pathlib import Path
import unittest

import torch

from conditional_drifting.sspa_update_screen import clone_adam, coordinate_scale, load_official, official_field, panel_moments


class UpdateScreenTests(unittest.TestCase):
    def test_clone_does_not_mutate_optimizer_parent(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        model(torch.ones(3, 2)).sum().backward()
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        clone = clone_adam(model, state, zero_first_moment=True)
        for original, modified in zip(state["state"].values(), clone.state.values()):
            self.assertGreater(original["exp_avg"].abs().sum(), 0)
            self.assertEqual(modified["exp_avg"].abs().sum(), 0)
            torch.testing.assert_close(original["exp_avg_sq"], modified["exp_avg_sq"])
            torch.testing.assert_close(original["step"], modified["step"])
        self.assertEqual(clone.param_groups[0]["lr"], .003)

    def test_normalized_epsilon_keeps_physical_cost(self):
        torch.manual_seed(4)
        g, p, r = [torch.randn(3, 4, 2, dtype=torch.float64) for _ in range(3)]
        scale = coordinate_scale(g, p, r)
        epsilon = .4
        reg = epsilon / scale.square()
        torch.testing.assert_close(.5 * torch.cdist(g / scale, p / scale).square() / reg,
                                   .5 * torch.cdist(g, p).square() / epsilon)

    def test_variance_gradient_matches_difference(self):
        scale = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
        cloud = torch.arange(24, dtype=torch.float64).reshape(2, 6, 2)
        truth = torch.zeros(2, 2, dtype=torch.float64)
        value = panel_moments(scale * cloud, truth)["variance"]
        derivative, = torch.autograd.grad(value, scale)
        plus = panel_moments((scale.detach() + 1e-5) * cloud, truth)["variance"]
        minus = panel_moments((scale.detach() - 1e-5) * cloud, truth)["variance"]
        torch.testing.assert_close(derivative, (plus - minus) / 2e-5)

    def test_scaled_loss_displacement_recovery(self):
        prediction = torch.randn(2, 4, 3, requires_grad=True)
        scale = .3
        displacement = torch.randn_like(prediction)
        target = (prediction / scale + displacement / scale).detach()
        loss = ((prediction / scale - target) ** 2).mean()
        gradient, = torch.autograd.grad(loss, prediction)
        recovered = -.5 * prediction.numel() * scale ** 2 * gradient
        torch.testing.assert_close(recovered, displacement)

    def test_official_loss_scaling_when_checkout_available(self):
        repo = Path(__file__).resolve().parents[1] / "results/official_sources_20261010/W-Flow"
        if not repo.exists():
            self.skipTest("Optional pinned official checkout is not present")
        official = load_official(repo)
        torch.manual_seed(42)
        prediction = torch.randn(3, 4, 2, requires_grad=True)
        positive, reference = torch.randn_like(prediction), torch.randn_like(prediction)
        scale = coordinate_scale(prediction.detach(), positive, reference)
        epsilon = .4
        R = epsilon / (scale.square().item() * prediction.shape[-1])
        loss, _ = official.drift_loss_ot(prediction, positive, reference, R_list=(R,),
            sinkhorn_num_iter=10, disable_diag_mask=True, use_quadratic_cost=True)
        gradient, = torch.autograd.grad(loss.mean(), prediction)
        raw, _, _ = official_field(official, prediction.detach() / scale,
                                   positive / scale, reference / scale, R * prediction.shape[-1])
        normalized = raw / raw.square().mean().clamp_min(1e-8).sqrt()
        expected = -2 * normalized / (scale * prediction.numel())
        torch.testing.assert_close(gradient, expected, atol=1e-6, rtol=1e-5)
