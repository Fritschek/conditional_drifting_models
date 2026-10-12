import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from conditional_drifting.baselines.fast_channels import physical_channel
from conditional_drifting.baselines.fast_evaluation import train_codec
from scripts.run_fast_channel_application import gaussian_law, reference_gradients, compare_gradients


class ApplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sspa_moments(self):
        torch.manual_seed(11)
        x = torch.randn(16, 8)
        mean, variance = gaussian_law("SSPA", x)
        samples = physical_channel("SSPA", x.repeat_interleave(8192, 0)).reshape(16, 8192, 8)
        self.assertLess((samples.mean(1)-mean).square().mean().sqrt().item(), 2*(variance/8192)**.5)
        self.assertAlmostEqual(samples.var(1).mean().item()/variance, 1., delta=.01)

    def test_codec_continuation_and_gradient_resolution(self):
        sampler = lambda x: physical_channel("SSPA", x)
        with tempfile.TemporaryDirectory() as temp:
            whole = train_codec(Path(temp)/"whole.pt", "SSPA", sampler, "cpu", 91, 6, 32, 3, False)
            train_codec(Path(temp)/"split.pt", "SSPA", sampler, "cpu", 91, 3, 32, 3, False)
            split = train_codec(Path(temp)/"split.pt", "SSPA", sampler, "cpu", 91, 6, 32, 3, True)
            for first, second in zip(whole[:2], split[:2]):
                for key, value in first.state_dict().items():
                    torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
            args = SimpleNamespace(channel="SSPA", gradient_repeats=4, gradient_samples=8,
                surrogate_gradient_repeats=2, eval_batch=128)
            reference = reference_gradients(*split[:3], args, 5)
            result = compare_gradients(sampler, split, reference, args, 12)
            for space in ("code", "encoder"):
                self.assertTrue(torch.isfinite(reference[space]["mean"]).all())
                self.assertGreater(reference[space]["mc_se_norm"], 0)
                self.assertIsInstance(reference[space]["resolved"], bool)
                self.assertGreater(result[space]["alignment"]["cosine"], .9)


if __name__ == "__main__":
    unittest.main()
