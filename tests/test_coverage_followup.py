from copy import deepcopy
from pathlib import Path
import os
import tempfile
import unittest

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer
from conditional_drifting.baselines.fast_evaluation import codec, evaluate_codec
from conditional_drifting.baselines.official_sources import load_sources
from conditional_drifting.sspa_update_screen import load_official
from scripts.coverage_followup_common import (MeasuredCoverageTrainer, CommonNoiseImplant,
    paired_curves, finish_generator, cached, record)
from scripts.run_sspa_input_coverage import CoverageTrainer
from scripts.run_sspa_coverage_replication import plan

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("FAST_TEST_DEVICE", "cpu")


class FollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.official = load_sources(ROOT / "results/official_fast_sources_20261012")
        cls.wflow = load_official(ROOT / "results/official_sources_20261010/W-Flow")

    def test_plan_counts_and_independent_seeds(self):
        import json
        parent = json.loads((ROOT / "results/fast_sspa_application_20261012/manifest.json").read_text())
        p = plan("paper", parent)
        self.assertEqual((p["generator_tasks"], p["surrogate_ae_tasks"], p["analytic_ae_tasks"]), (12, 36, 9))
        self.assertEqual(p["training_oracle_outputs"], 491520000)
        self.assertEqual(len(set(sum(p["ae_seeds"].values(), []))), 9)
        self.assertNotIn(9111, p["generator_seeds"])
        self.assertTrue(set(plan("smoke", parent)["generator_seeds"]).isdisjoint(p["generator_seeds"]))

    def test_energy_accounting_preserves_updates_and_resume(self):
        for family in ("gaussian", "sinkhorn"):
            cfg = FastConfig(channel="SSPA", family=family, parameters=1024, batch_size=128, updates=3)
            old = CoverageTrainer(cfg, DEVICE, self.official, self.wflow)
            expected = [old.step() for _ in range(3)]
            new = MeasuredCoverageTrainer(cfg, DEVICE, self.official, self.wflow)
            self.assertEqual(expected, [new.step() for _ in range(3)])
            self.assertGreater(new.input_energy_sum, 0)
            state = deepcopy(new.state())
            row = new.step()
            resumed = MeasuredCoverageTrainer(cfg, DEVICE, self.official, self.wflow)
            resumed.restore(state)
            self.assertEqual(row, resumed.step())
            self.assertEqual(new.input_energy_sum, resumed.input_energy_sum)
            for key, value in new.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)

    def test_completed_generator_not_rewritten(self):
        cfg = FastConfig(channel="SSPA", family="gaussian", parameters=1024, batch_size=128, updates=1)
        trainer = MeasuredCoverageTrainer(cfg, DEVICE, self.official, self.wflow)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generator.pt"
            finish_generator(trainer, path, 1, False)
            before = path.read_bytes()
            fresh = MeasuredCoverageTrainer(cfg, DEVICE, self.official, self.wflow)
            finish_generator(fresh, path, 1, True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(fresh.update, 1)

    def test_paired_evaluation_and_identical_codec(self):
        e, d, messages = codec("SSPA", DEVICE, 192010)
        trained = (e, d, messages, 0.)
        values = paired_curves(dict(a=trained, b=trained), 8, 128, 45)
        original = evaluate_codec("SSPA", e, d, messages, 8, 128, 45, [-4, -2, 0])
        for curve, ref in zip(values, original):
            p = curve["variants"]["a"]
            self.assertEqual(p["errors"], ref["errors"])
            self.assertEqual(p["bit_errors"], ref["bit_errors"])
            self.assertEqual(sum(p["errors_per_message"]), p["errors"])
            pair = curve["pairs"][0]
            self.assertEqual(pair["a_only_wrong"]+pair["b_only_wrong"], 0)
            self.assertEqual(pair["pooled_paired_mc_se"], 0.)
        self.assertEqual(values, paired_curves(dict(a=trained, b=trained), 8, 128, 45))

    def test_common_noise_ignores_internal_rng(self):
        cfg = FastConfig(family="diffusion", channel="SSPA", parameters=1024)
        trainer = FastTrainer(cfg, DEVICE, self.official)
        implant = CommonNoiseImplant(trainer, 20).reset(41)
        x = torch.ones(4, 8, device=DEVICE)
        def noisy_sample(x, steps, noise, model):
            torch.randn(steps*10, device=x.device)
            return noise
        trainer.sample = noisy_sample
        a = [implant(x).detach() for _ in range(2)]
        implant.steps = 100
        implant.reset(41)
        b = [implant(x).detach() for _ in range(2)]
        for u, v in zip(a, b):
            torch.testing.assert_close(u, v, rtol=0, atol=0)
        self.assertFalse(torch.equal(a[0], a[1]))

    def test_task_stamp_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            record(path, dict(checkpoint="abc"), dict(done=True))
            self.assertEqual(cached(path, dict(checkpoint="abc"), True), dict(done=True))
            with self.assertRaises(ValueError):
                cached(path, dict(checkpoint="changed"), True)
            with self.assertRaises(FileExistsError):
                cached(path, dict(checkpoint="abc"), False)

    def test_sampler_grids_execute_declared_nfe_and_input_backward(self):
        from scripts.run_saved_sampler_panel import GRIDS
        for family, grid in GRIDS.items():
            trainer = FastTrainer(FastConfig(channel="SSPA", family=family, parameters=1024),
                                  DEVICE, self.official)
            for nfe in grid:
                implant = CommonNoiseImplant(trainer, nfe).reset(46)
                calls = []
                hook = implant.model.register_forward_hook(lambda *args: calls.append(1))
                x = torch.ones(2, 8, device=DEVICE, requires_grad=True)
                y = implant(x)
                self.assertEqual(len(calls), nfe)
                self.assertTrue(torch.isfinite(torch.autograd.grad(y.square().mean(), x)[0]).all())
                hook.remove()


if __name__ == "__main__":
    unittest.main()
