import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from conditional_drifting.sspa_intervention import BRANCHES, Intervention, digest, passes_gate
from conditional_drifting.sspa_trajectory import SSPATrajectory, TrajectoryConfig, rng_state
from conditional_drifting.sspa_update_screen import load_official

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sspa_interventions import equal_state


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = os.environ.get("TRAJECTORY_TEST_DEVICE", "cpu")
        cls.official = load_official(Path("results/official_sources_20261010/W-Flow"))

    def parent(self, directory):
        cfg = TrajectoryConfig(policy="fixed_common", batch_size=16, hidden_dim=16,
            calibration_anchors=8, validation_anchors=4, validation_samples=8, validation_projections=4)
        trainer = SSPATrajectory(cfg, self.device)
        trainer.step()
        trainer.validate()
        path = Path(directory) / "parent.pt"
        trainer.save(path)
        return trainer, path

    def test_control_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            old, path = self.parent(directory)
            for _ in range(3):
                old.step(True)
            expected_rng = rng_state()
            child = Intervention(path, "control", self.device, self.official, {})
            child.validate_child()
            for _ in range(3):
                child.step(True)
            self.assertTrue(equal_state(old.model.state_dict(), child.model.state_dict()))
            self.assertTrue(equal_state(old.optimizer.state_dict(), child.optimizer.state_dict()))
            self.assertTrue(equal_state(expected_rng, rng_state()))

    def test_all_branches_resume_and_parent_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            _, parent = self.parent(directory)
            before = digest(parent)
            for branch in BRANCHES:
                with self.subTest(branch=branch):
                    a = Intervention(parent, branch, self.device, self.official, {"version": 1})
                    a.step(True)
                    checkpoint = Path(directory) / "child.pt"
                    a.save_child(checkpoint)
                    a.step(True)
                    expected_rng = rng_state()
                    b = Intervention(parent, branch, self.device, self.official, {"version": 1})
                    b.load_child(checkpoint)
                    b.validate_child()
                    b.step(True)
                    self.assertTrue(equal_state(a.model.state_dict(), b.model.state_dict()))
                    self.assertTrue(equal_state(a.optimizer.state_dict(), b.optimizer.state_dict()))
                    self.assertTrue(equal_state(expected_rng, rng_state()))
                    if a.ema is not None:
                        self.assertTrue(equal_state(a.ema.state_dict(), b.ema.state_dict()))
                    bad = Intervention(parent, branch, self.device, self.official, {"version": 2})
                    with self.assertRaisesRegex(ValueError, "contract"):
                        bad.load_child(checkpoint)
            self.assertEqual(before, digest(parent))

    def test_interventions_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            _, parent = self.parent(directory)
            beta = Intervention(parent, "log10_beta0", self.device, self.official, {})
            beta.step()
            self.assertEqual(beta.optimizer.param_groups[0]["betas"], (0., .999))
            lr = Intervention(parent, "log10_lr01", self.device, self.official, {})
            self.assertEqual(lr.optimizer.param_groups[0]["lr"], .0001)
            drift = Intervention(parent, "log10_drift01", self.device, self.official, {})
            self.assertEqual(drift.cfg.drift_scale, .1)
            official = Intervention(parent, "official64", self.device, self.official, {})
            self.assertEqual(len(official.optimizer.state), 0)
            self.assertEqual(official.cfg.samples * official.cfg.batch_size, 64)
            official.step()
            self.assertEqual(official.optimizer.param_groups[0]["lr"], 1e-6)

    def test_gate_requires_all_three_points(self):
        parent = dict(anchor_swd=.04, anchor_swd_floor=.02, anchor_mean_l2=.03, anchor_mean_l2_floor=.02)
        rows = [dict(update=u, anchor_swd=.05, conditional_variance=.05, anchor_mean_l2=.04)
                for u in (180000, 190000, 200000)]
        self.assertTrue(passes_gate(rows, parent, .32))
        self.assertFalse(passes_gate(rows[:2], parent, .32))
        self.assertFalse(passes_gate(rows, parent, .32, ema=True))
        rows[-1]["conditional_variance"] = 1.
        self.assertFalse(passes_gate(rows, parent, .32))

    def test_adam_gradient_scaling_is_not_learning_rate_scaling(self):
        # Exact limiting case: fresh moments, epsilon=0, no clipping/decay.
        values = [torch.ones(2, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        optimizers = [torch.optim.Adam([value], lr=lr, eps=0.)
                      for value, lr in zip(values, (.001, .001, .0001))]
        for step in range(10):
            gradient = torch.tensor([1. + step, 2. + step], dtype=torch.float64)
            for value, optimizer, scale in zip(values, optimizers, (1., .1, 1.)):
                value.grad = scale * gradient
                optimizer.step()
        torch.testing.assert_close(values[0], values[1], rtol=0., atol=1e-14)
        torch.testing.assert_close(1. - values[2], .1 * (1. - values[0]), rtol=0., atol=1e-14)
