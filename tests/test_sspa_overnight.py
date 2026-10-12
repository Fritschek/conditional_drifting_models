import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_sspa_interventions import equal_state
from run_sspa_overnight import child, selected_update, late_gate, Budget, BudgetExpired
from conditional_drifting.sspa_trajectory import SSPATrajectory, TrajectoryConfig, rng_state
from conditional_drifting.sspa_update_screen import load_official
from conditional_drifting.sspa_utility import CodecTrial, analytic_channel, load_generator, NOISE_STD
from scripts.run_local_gradient_fidelity import normalized_codebook


class FreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = os.environ.get("TRAJECTORY_TEST_DEVICE", "cpu")
        cls.official = load_official(ROOT / "results/official_sources_20261010/W-Flow")

    def parent(self, folder):
        cfg = TrajectoryConfig(seed=9904, policy="fixed_common", batch_size=16,
            calibration_anchors=8, validation_anchors=4, validation_samples=8, validation_projections=4)
        trainer = SSPATrajectory(cfg, self.device)
        before = rng_state()
        trainer.validate()
        self.assertTrue(equal_state(before, rng_state()))
        path = Path(folder) / "parent.pt"
        trainer.save(path)
        self.assertEqual(trainer.update, 0)
        self.assertFalse(trainer.optimizer.state)
        return path

    def test_fresh_shared_initialization_exposure_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = self.parent(folder)
            initial = torch.load(parent, weights_only=False, map_location="cpu")
            for variant in ("old_k4", "old_k64", "official64"):
                with self.subTest(variant=variant):
                    a = child(parent, variant, self.device, self.official, {"variant": variant})
                    self.assertTrue(equal_state({k: v.cpu() for k, v in a.model.state_dict().items()}, initial["model"]))
                    self.assertFalse(a.optimizer.state)
                    self.assertEqual(a.cfg.batch_size * a.cfg.samples, 64)
                    self.assertEqual(a.cfg.samples, 4 if variant == "old_k4" else 64)
                    a.step(True)
                    self.assertEqual(a.counts["training_generated_outputs"], 64)
                    path = Path(folder) / "child.pt"
                    a.save_child(path)
                    a.step(True)
                    expected_rng = rng_state()
                    b = child(parent, variant, self.device, self.official, {"variant": variant})
                    b.load_child(path)
                    b.validate_child()
                    b.step(True)
                    for name in ("model", "optimizer"):
                        self.assertTrue(equal_state(getattr(a, name).state_dict(), getattr(b, name).state_dict()))
                    self.assertTrue(equal_state(expected_rng, rng_state()))
                    if a.ema is not None:
                        self.assertTrue(equal_state(a.ema.state_dict(), b.ema.state_dict()))

    def test_codec_resume_validation_counts_and_power(self):
        with tempfile.TemporaryDirectory() as folder:
            a = CodecTrial(19904, analytic_channel, self.device, {})
            codes = normalized_codebook(a.encoder, a.messages)
            self.assertLess(abs(codes.mean().item()), 1e-6)
            self.assertAlmostEqual(codes.square().mean().item(), 1., places=6)
            a.step(True)
            path = Path(folder) / "codec.pt"
            a.save(path)
            before = rng_state()
            label_before = a.labels.get_state()
            stats = a.evaluate(8, 123)
            self.assertTrue(equal_state(before, rng_state()))
            self.assertTrue(torch.equal(label_before, a.labels.get_state()))
            self.assertEqual(stats["trials"], 64 * 8)
            self.assertEqual(stats["bit_trials"], 6 * stats["trials"])
            self.assertEqual(stats["ser"], stats["errors"] / stats["trials"])
            self.assertEqual(stats["ber"], stats["bit_errors"] / stats["bit_trials"])
            a.step(True)
            expected_rng = rng_state()
            b = CodecTrial(19904, analytic_channel, self.device, {})
            b.load(path)
            b.step(True)
            for name in ("encoder", "decoder", "optimizer"):
                self.assertTrue(equal_state(getattr(a, name).state_dict(), getattr(b, name).state_dict()))
            self.assertTrue(equal_state(expected_rng, rng_state()))
            self.assertTrue(torch.equal(a.labels.get_state(), b.labels.get_state()))

    def test_paired_labels_despite_different_channel_rng(self):
        a = CodecTrial(19904, analytic_channel, self.device, {})
        initial = {k: v.clone() for k, v in a.encoder.state_dict().items()}
        for _ in range(3):
            a.step()
        label_state = a.labels.get_state()

        def extra_draws(x):
            torch.randn(117, device=x.device)
            return analytic_channel(x)

        b = CodecTrial(19904, extra_draws, self.device, {})
        self.assertTrue(equal_state(initial, b.encoder.state_dict()))
        for _ in range(3):
            b.step()
        self.assertTrue(torch.equal(label_state, b.labels.get_state()))

    def test_frozen_direct_generator_allows_input_gradient(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = self.parent(folder)
            model = load_generator(parent, self.device)
            x = torch.randn(11, 8, device=self.device, requires_grad=True)
            model(x).sum().backward()
            self.assertGreater(x.grad.norm().item(), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))

    def test_selection_gate_and_deadline(self):
        rows = [dict(update=0, anchor_swd=0), dict(update=20, anchor_swd=1), dict(update=10, anchor_swd=1)]
        self.assertEqual(selected_update(rows), 10)
        valid = [dict(update=u, anchor_swd=.02, anchor_swd_floor=.015,
            anchor_mean_l2=.04, anchor_mean_l2_floor=.03, conditional_variance=NOISE_STD ** 2 / 2)
            for u in (300000, 350000, 390720)]
        self.assertTrue(late_gate(valid))
        self.assertFalse(late_gate(valid[:2]))
        valid[-1]["anchor_swd"] = 1.
        self.assertFalse(late_gate(valid))
        with self.assertRaises(BudgetExpired):
            Budget("2000-01-01T00:00:00+00:00").check()


if __name__ == "__main__":
    unittest.main()
