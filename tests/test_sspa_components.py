import os
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from conditional_drifting.sspa_components import ComponentIntervention, VARIANTS, component_loss
from conditional_drifting.sspa_intervention import Intervention, digest
from conditional_drifting.sspa_trajectory import SSPATrajectory, TrajectoryConfig, rng_state
from conditional_drifting.sspa_update_screen import load_official

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sspa_interventions import equal_state


class ComponentTests(unittest.TestCase):
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
        return path

    def test_official_step_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.parent(directory)
            original = Intervention(parent, "official64", self.device, self.official, {})
            for _ in range(3):
                original.step()
            expected_rng = rng_state()
            new = ComponentIntervention(parent, "official64", self.device, self.official, {})
            for _ in range(3):
                new.step()
            for name in ("model", "optimizer", "ema"):
                self.assertTrue(equal_state(getattr(original, name).state_dict(), getattr(new, name).state_dict()))
            self.assertTrue(equal_state(expected_rng, rng_state()))

    def test_raw_loss_gradient_and_epsilon_conversion(self):
        torch.manual_seed(53)
        g = torch.randn(3, 8, 8, device=self.device, requires_grad=True)
        p, r = torch.randn_like(g), torch.randn_like(g)
        normalized, scale, force, _ = component_loss(self.official, g, p, r, "official64", .3887)
        a = torch.autograd.grad(normalized, g)[0]
        raw, _, _, _ = component_loss(self.official, g, p, r, "raw_loss64", .3887)
        b = torch.autograd.grad(raw, g)[0]
        torch.testing.assert_close(b, a * scale.square() * force.clamp_min(1e-8).sqrt(), rtol=1e-4, atol=2e-7)
        _, scale, _, reg = component_loss(self.official, g, p, r, "fixed_epsilon64", .3887)
        self.assertAlmostEqual(reg * 8 * scale.square().item(), .3887)

    def test_component_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.parent(directory)
            before = digest(parent)
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    contract = {"component": variant}
                    a = ComponentIntervention(parent, variant, self.device, self.official, contract)
                    self.assertEqual(a.cfg.samples, 4 if variant == "k4" else 64)
                    self.assertEqual(a.cfg.samples * a.cfg.batch_size, 64)
                    self.assertEqual(bool(a.optimizer.state), variant == "old_optimizer64")
                    self.assertEqual(a.optimizer.param_groups[0]["betas"], (.9, .999) if variant == "old_optimizer64" else (.9, .95))
                    a.step()
                    path = Path(directory) / "child.pt"
                    a.save_child(path)
                    a.step(True)
                    expected_rng = rng_state()
                    b = ComponentIntervention(parent, variant, self.device, self.official, contract)
                    b.load_child(path)
                    b.validate_child()
                    b.step(True)
                    for name in ("model", "optimizer", "ema"):
                        self.assertTrue(equal_state(getattr(a, name).state_dict(), getattr(b, name).state_dict()))
                    self.assertTrue(equal_state(expected_rng, rng_state()))
                    wrong = ComponentIntervention(parent, variant, self.device, self.official, contract | {"version": 2})
                    with self.assertRaisesRegex(ValueError, "contract"):
                        wrong.load_child(path)
            self.assertEqual(before, digest(parent))

    def test_particle_bridge_keeps_old_optimizer_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = self.parent(directory)
            old = torch.load(parent, map_location="cpu", weights_only=False)
            a = Intervention(parent, "control", self.device, None, {"component": "production_k64"})
            a.cfg = replace(a.cfg, batch_size=1, samples=64)
            self.assertTrue(equal_state(a.optimizer.state_dict(), old["optimizer"]))
            self.assertIsNone(a.ema)
            a.step(True)
            path = Path(directory) / "bridge.pt"
            a.save_child(path)
            a.step(True)
            expected_rng = rng_state()
            b = Intervention(parent, "control", self.device, None, {"component": "production_k64"})
            b.cfg = replace(b.cfg, batch_size=1, samples=64)
            b.load_child(path)
            b.validate_child()
            b.step(True)
            self.assertTrue(equal_state(a.model.state_dict(), b.model.state_dict()))
            self.assertTrue(equal_state(a.optimizer.state_dict(), b.optimizer.state_dict()))
            self.assertTrue(equal_state(expected_rng, rng_state()))


if __name__ == "__main__":
    unittest.main()
