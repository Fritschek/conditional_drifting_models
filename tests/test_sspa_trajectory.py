import dataclasses
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from conditional_drifting.channels import sspa
from conditional_drifting.losses import compute_fiber_sinkhorn_drift
from conditional_drifting.sspa_trajectory import (SSPATrajectory, TrajectoryConfig,
                                                field, isolated_rng)
from conditional_drifting.training import DriftingConfig, set_seed, train_conditional_drifting


class TrajectoryTests(unittest.TestCase):
    device = os.environ.get("TRAJECTORY_TEST_DEVICE", "cpu")

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **kwargs):
        return TrajectoryConfig(batch_size=8, hidden_dim=16, calibration_anchors=8,
                                validation_anchors=4, validation_samples=8,
                                validation_projections=4, **kwargs)

    def assert_model_equal(self, a, b):
        for name, value in a.model.state_dict().items():
            self.assertTrue(torch.equal(value, b.model.state_dict()[name]), name)

    def test_exact_resume(self):
        for policy in ("legacy_separate_adaptive", "fixed_common", "shared_adaptive"):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as d:
                cfg = self.config(policy=policy)
                a = SSPATrajectory(cfg, self.device)
                for _ in range(5):
                    a.step(True)
                b = SSPATrajectory(cfg, self.device)
                for _ in range(2):
                    b.step(True)
                path = Path(d) / "checkpoint.pt"
                b.save(path)
                b = SSPATrajectory(cfg, self.device)
                b.load(path)
                for _ in range(3):
                    b.step(True)
                self.assert_model_equal(a, b)
                self.assertEqual(a.trace, b.trace)
                self.assertEqual(a.counts, b.counts)
                self.assertEqual(b.update, 5)

    def test_validation_does_not_change_training(self):
        a = SSPATrajectory(self.config(), self.device)
        for _ in range(4):
            a.step(True)
        b = SSPATrajectory(self.config(), self.device)
        b.validate()
        for _ in range(4):
            b.step(True)
            b.validate()
        self.assert_model_equal(a, b)
        self.assertEqual(a.trace, b.trace)
        self.assertEqual(b.counts["training_anchors"], 32)
        self.assertEqual(b.counts["training_oracle_outputs"], 128)
        self.assertEqual(b.counts["training_generated_outputs"], 128)
        self.assertEqual(b.counts["training_reference_outputs"], 128)
        self.assertEqual(b.counts["calibration_oracle_outputs"], 64)
        self.assertEqual(b.counts["validation_oracle_outputs"], 320)
        self.assertEqual(b.counts["diagnostic_oracle_outputs"], 80)

    def test_legacy_matches_original_training(self):
        cfg = self.config()
        a = SSPATrajectory(cfg, self.device)
        for _ in range(4):
            a.step()
        old = DriftingConfig(n=8, noise_std=cfg.noise_std, batch_size=8, dataset_size=16,
                             epochs=2, hidden_dim=16, drift_field="fiber_sinkhorn", is_residual=False)
        set_seed(cfg.seed)
        model, _ = train_conditional_drifting(sspa, old, torch.device(self.device))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, a.model.state_dict()[name]), name)

    def test_field_equivalence_and_common_epsilon(self):
        cfg = self.config()
        with isolated_rng(17):
            g, p, r = [torch.randn(32, 8, device=self.device) for _ in range(3)]
        old = compute_fiber_sinkhorn_drift(g, p, generated_reference=r, fiber_num_conditions=8,
            fiber_generated_samples=4, fiber_positive_samples=4, fiber_reference_samples=4, max_drift_norm=2.)
        actual, _ = field(g, p, r, cfg, .5, True)
        self.assertTrue(torch.equal(old, actual))
        for policy in ("fixed_common", "shared_adaptive"):
            _, info = field(g, p, r, dataclasses.replace(cfg, policy=policy), .5, True)
            self.assertEqual(info["cross_epsilon"], info["self_epsilon"])
        fixed, _ = field(g, p, r, dataclasses.replace(cfg, policy="fixed_common"), .5)
        old_fixed = compute_fiber_sinkhorn_drift(g, p, generated_reference=r, fiber_num_conditions=8,
            fiber_generated_samples=4, fiber_positive_samples=4, fiber_reference_samples=4,
            max_drift_norm=2., sinkhorn_epsilon=.5)
        self.assertTrue(torch.equal(fixed, old_fixed))

    def test_reject_mismatched_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.pt"
            a = SSPATrajectory(self.config(), self.device)
            a.save(path)
            b = SSPATrajectory(self.config(seed=9002), self.device)
            with self.assertRaisesRegex(ValueError, "configuration"):
                b.load(path)
            state = torch.load(path, weights_only=False)
            state["source_hashes"]["losses.py"] = "changed"
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "source"):
                a.load(path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
