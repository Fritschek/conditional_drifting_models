from copy import deepcopy
import os
from pathlib import Path
import unittest

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer
from conditional_drifting.baselines.official_sources import load_sources
from conditional_drifting.sspa_update_screen import load_official
from scripts.run_sspa_input_coverage import CoverageTrainer

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("FAST_TEST_DEVICE", "cpu")


class CoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.official = load_sources(ROOT/"results/official_fast_sources_20261012")
        cls.wflow = load_official(ROOT/"results/official_sources_20261010/W-Flow")

    def test_control_matches_frozen_updates(self):
        for family in ("gaussian", "sinkhorn"):
            cfg = FastConfig(channel="SSPA", family=family, batch_size=128, updates=10)
            base = FastTrainer(cfg, DEVICE, self.official, self.wflow)
            expected = [base.step() for _ in range(10)]
            state = deepcopy(base.state())
            candidate = CoverageTrainer(cfg, DEVICE, self.official, self.wflow, law="normal")
            self.assertEqual(expected, [candidate.step() for _ in range(10)])
            self.assertEqual(state["counts"], candidate.counts)
            for part in ("model", "ema"):
                for key, value in state[part].items():
                    torch.testing.assert_close(value, candidate.state()[part][key], rtol=0, atol=0)

    def test_broad_resume(self):
        for family in ("gaussian", "sinkhorn"):
            cfg = FastConfig(channel="SSPA", family=family, batch_size=128, updates=10)
            a = CoverageTrainer(cfg, DEVICE, self.official, self.wflow)
            a.step()
            saved = deepcopy(a.state())
            expected = a.step()
            b = CoverageTrainer(cfg, DEVICE, self.official, self.wflow)
            b.restore(saved)
            self.assertEqual(expected, b.step())
            self.assertEqual(a.broad_anchors, b.broad_anchors)
            for key, value in a.model.state_dict().items():
                torch.testing.assert_close(value, b.model.state_dict()[key], rtol=0, atol=0)

    def test_mixture_sampling(self):
        cfg = FastConfig(channel="SSPA", family="gaussian")
        a = CoverageTrainer(cfg, DEVICE, self.official, self.wflow)
        values = a.inputs(100000)
        self.assertAlmostEqual(a.broad_anchors/len(values), .25, delta=.006)
        self.assertAlmostEqual((values.norm(dim=1)>10).float().mean().item(), .25*(1-10/a.radius_max), delta=.006)
        self.assertLess(values.mean(0).abs().max().item(), .05)


if __name__ == "__main__":
    unittest.main()
