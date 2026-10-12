import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer
from conditional_drifting.baselines.official_sources import load_sources, definitions
from scripts.run_fast_baseline_development import moments
from scripts.run_consistency_warmstart_development import loss_step

ROOT = Path(__file__).resolve().parents[1]


class DevelopmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.official = load_sources(ROOT / "results/official_fast_sources_20261012")
        namespace = dict(vars(cls.official.cm))
        definitions(ROOT / "results/official_fast_sources_20261012/consistency_models/cm/resample.py",
                    ("LogNormalSampler",), namespace)
        cls.sigma_sampler = namespace["LogNormalSampler"]()

    def test_known_awgn_moments(self):
        trainer = SimpleNamespace(n=7, device="cpu", spec={"noise_std": .5},
            sample=lambda x, nfe, model: x + .5*torch.randn_like(x))
        result = moments(trainer, None, 1, 42)
        self.assertAlmostEqual(result["variance_ratio"], 1., delta=.02)
        self.assertAlmostEqual(result["mean_rmse"], result["mean_mc_rmse_floor"], delta=.004)

    def test_edm_and_distillation_restart(self):
        for kind in ("edm", "cd"):
            with self.subTest(kind=kind):
                cfg = FastConfig(channel="AWGN", family="consistency", batch_size=32, updates=10)
                trainer = FastTrainer(cfg, "cpu", self.official)
                teacher = copy.deepcopy(trainer.model).requires_grad_(False)
                denoiser = self.official.cm.KarrasDenoiser(distillation=False, loss_norm="l2")
                loss_step(trainer, kind, denoiser, self.sigma_sampler, teacher)
                checkpoint = copy.deepcopy(trainer.state())
                first = loss_step(trainer, kind, denoiser, self.sigma_sampler, teacher)
                weights = copy.deepcopy(trainer.model.state_dict())
                restored = FastTrainer(cfg, "cpu", self.official)
                restored.restore(checkpoint)
                second = loss_step(restored, kind, denoiser, self.sigma_sampler, teacher)
                self.assertEqual(first, second)
                for name, value in weights.items():
                    torch.testing.assert_close(value, restored.model.state_dict()[name], atol=0, rtol=0)
                self.assertEqual(restored.counts["oracle_outputs"], 64)
                if kind == "cd":
                    self.assertEqual(restored.counts["teacher_network_outputs"], 128)
                    self.assertEqual(restored.counts["target_network_outputs"], 64)
                self.assertTrue(all(p.grad is None for p in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
