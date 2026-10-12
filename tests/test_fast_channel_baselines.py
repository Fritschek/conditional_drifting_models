"""Numerical and restart checks against pinned upstream routines."""

import copy
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from conditional_drifting.baselines.fast_channels import (FastConfig, FastTrainer, FastImplant,
    GaussianNetwork, FAMILIES, channel_spec, physical_channel, parameter_count)
from conditional_drifting.baselines.official_sources import load_sources
from conditional_drifting.baselines.fast_evaluation import distribution_metrics, train_codec, save
from conditional_drifting.baselines.fast_checkpoint import load_fast_implant
from conditional_drifting.sspa_update_screen import load_official
from conditional_drifting.sspa_trajectory import isolated_rng
from scripts.run_fast_channel_comparison import ensure_manifest, source_files

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("FAST_TEST_DEVICE", "cpu")


class FastBaselinesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.official = load_sources(ROOT / "results/official_fast_sources_20261012")
        cls.wflow = load_official(ROOT / "results/official_sources_20261010/W-Flow")

    def trainer(self, family, **kw):
        cfg = replace(FastConfig(family=family, channel="AWGN", batch_size=128, updates=20), **kw)
        return FastTrainer(cfg, DEVICE, self.official, self.wflow)

    def test_upstream_flow_path(self):
        x0, x1 = torch.randn(4, 7), torch.randn(4, 7)
        t = torch.tensor([0., .25, .5, 1.])
        path = self.official.path.sample(x_0=x0, x_1=x1, t=t)
        torch.testing.assert_close(path.x_t, (1-t[:, None])*x0+t[:, None]*x1)
        torch.testing.assert_close(path.dx_t, x1-x0)

    def test_consistency_boundary(self):
        trainer = self.trainer("consistency")
        x = torch.randn(4, 7, device=DEVICE)
        sigma = x.new_full((4,), trainer.cfg.consistency_sigma_min)
        _, y = trainer.consistency.denoise(trainer.model, x, sigma, condition=x)
        torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)
        trainer.step()
        self.assertTrue(all(p.grad is None for p in trainer.target.parameters()))

    def test_all_families_capacity_counts_and_resume(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                trainer = self.trainer(family)
                initial = 9 if family == "wgan" else 4
                for _ in range(initial):
                    trainer.step()
                checkpoint = copy.deepcopy(trainer.state())
                first = trainer.step()
                weights = copy.deepcopy(trainer.state())
                restored = self.trainer(family)
                restored.restore(checkpoint)
                second = restored.step()
                self.assertEqual(first, second)
                for kind in ("model", "ema", "target", "critic"):
                    if weights[kind] is not None:
                        for key, value in weights[kind].items():
                            torch.testing.assert_close(value, restored.state()[kind][key], rtol=0, atol=0)
                self.assertEqual(restored.counts["oracle_outputs"], (initial+1)*128)
                self.assertEqual(restored.counts["input_anchors"], (initial+1)*(2 if family == "sinkhorn" else 128))
                self.assertEqual(restored.counts["generator_updates"], 2 if family == "wgan" else 5)
                self.assertLess(abs(parameter_count(trainer.model)/60000-1), .1)

    def test_sampler_nfe_and_input_derivatives(self):
        for family in FAMILIES:
            for steps in ({"flow": [1,2,4,8,16], "consistency": [1,2], "diffusion": [10,20,50,100]}.get(family, [1])):
                with self.subTest(family=family, steps=steps):
                    trainer = self.trainer(family)
                    implant = FastImplant(trainer, steps)
                    calls = []
                    hook = implant.model.register_forward_hook(lambda *a: calls.append(1))
                    x = torch.randn(2, 7, device=DEVICE, requires_grad=True)
                    with isolated_rng(93):
                        y = implant(x)
                        grad = torch.autograd.grad(y.mean(), x)[0]
                    self.assertEqual(len(calls), steps)
                    hook.remove()
                    self.assertTrue(torch.isfinite(y).all() and torch.isfinite(grad).all())
                    self.assertGreater(grad.norm().item(), 0.)
                    direction = torch.randn_like(x)
                    direction /= direction.norm()
                    h = .01
                    with isolated_rng(93), torch.no_grad():
                        plus = implant(x+h*direction).mean()
                    with isolated_rng(93), torch.no_grad():
                        minus = implant(x-h*direction).mean()
                    torch.testing.assert_close((plus-minus)/(2*h), (grad*direction).sum(), rtol=.04, atol=.002)
                    self.assertTrue(all(p.grad is None for p in implant.parameters()))

    def test_full_gaussian(self):
        model = GaussianNetwork(8, 60000, full=True).to(DEVICE)
        x = torch.randn(5, 8, device=DEVICE, requires_grad=True)
        _, scale = model.distribution(x)
        self.assertTrue((scale.diagonal(dim1=-2, dim2=-1)>0).all())
        model.loss(x, torch.randn_like(x)).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_all_channels_train_and_evaluate(self):
        for channel in ("AWGN", "Rayleigh", "SSPA", "TDL"):
            with self.subTest(channel=channel):
                trainer = self.trainer("flow", channel=channel)
                trainer.step()
                implant = FastImplant(trainer)
                before = torch.get_rng_state().clone()
                result = distribution_metrics(channel, implant, DEVICE, 9, 64, 4, 8, 8, 32)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertGreater(result["analytic_floor"]["swd"], 0)
                spec = channel_spec(channel)
                x = torch.randn(4, spec["n"], device=DEVICE, requires_grad=True)
                gradient = torch.autograd.grad(physical_channel(channel, x).sum(), x)[0]
                self.assertTrue(torch.isfinite(gradient).all())

    def test_codec_resume(self):
        trainer = self.trainer("consistency")
        implant = FastImplant(trainer, 2)
        with tempfile.TemporaryDirectory() as root:
            path1, path2 = Path(root)/"one.pt", Path(root)/"two.pt"
            one = train_codec(path1, "AWGN", implant, DEVICE, 54, 4, 32, 2, False)
            train_codec(path2, "AWGN", implant, DEVICE, 54, 2, 32, 2, False)
            two = train_codec(path2, "AWGN", implant, DEVICE, 54, 4, 32, 2, True)
            for a, b in zip(one[:2], two[:2]):
                for key, value in a.state_dict().items():
                    torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)

    def test_manifest_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"run/manifest.json"
            ensure_manifest(path, {"seed": 4}, False)
            ensure_manifest(path, {"seed": 4}, True)
            with self.assertRaises(ValueError):
                ensure_manifest(path, {"seed": 5}, True)
            with self.assertRaises(FileExistsError):
                ensure_manifest(path, {"seed": 4}, False)

    def test_checkpoint_implant(self):
        trainer = self.trainer("consistency")
        trainer.step()
        state = trainer.state()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = dict(adapter_files=source_files(), sources=self.official.records, configs=[state["config"]])
            ensure_manifest(root / "suite/manifest.json", manifest, False)
            path = root / "suite/AWGN/seed9101/consistency/generator.pt"
            save(path, dict(trainer=state))
            implant = load_fast_implant(path, ROOT / "results/official_fast_sources_20261012", device=DEVICE, steps=2)
            x = torch.randn(4, 7, device=DEVICE)
            with isolated_rng(99):
                a = trainer.sample(x, 2)
            with isolated_rng(99):
                b = implant(x)
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
