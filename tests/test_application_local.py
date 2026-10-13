from copy import deepcopy
import math
import os
from pathlib import Path
import tempfile
import unittest

import torch

from conditional_drifting.application_channels import (SSPACoverageChannel, OFDMChannel,
    as_complex, as_real, real_noise_variance, complex_normal)
from conditional_drifting.ofdm_application import Constellation, PilotReceiver, train_receiver, evaluate_receiver
from conditional_drifting.baselines.application_generators import FastTrainer, FastConfig, FastImplant
from conditional_drifting.baselines.official_sources import load_sources
from conditional_drifting.sspa_update_screen import load_official
from scripts.application_local_common import train_stage

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("FAST_TEST_DEVICE", "cpu")


class PhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sspa_independent_physical_calibration(self):
        c = SSPACoverageChannel()
        x = torch.ones(131072, 8, device=DEVICE)
        noise = c.sample(x)-c.mean(x)
        energy_per_bit = x.square().sum(-1).mean().item()/6
        measured_n0 = 2*noise.double().var().item()
        measured_db = 10*math.log10(energy_per_bit/measured_n0)
        self.assertAlmostEqual(measured_db, c.ebno_db, delta=.03)
        self.assertAlmostEqual(c.variance, .0528297730820, places=11)
        self.assertAlmostEqual(real_noise_variance(8/6, 8), 2*c.variance, places=12)

    def test_input_energy_and_pilots(self):
        g = torch.Generator(device=DEVICE).manual_seed(27)
        c = SSPACoverageChannel()
        x = c.inputs(50000, DEVICE, g)
        self.assertAlmostEqual(x.square().sum(-1).mean().item(), 146/3, delta=1.)
        c = OFDMChannel()
        x = as_complex(c.inputs(20000, DEVICE, g))
        torch.testing.assert_close(x[:, list(c.pilot_indices)], torch.ones_like(x[:, list(c.pilot_indices)]))
        self.assertAlmostEqual(x.abs().square().mean().item(), 1., delta=.01)

    def test_ofdm_noise_covariance_and_fractional_delays(self):
        torch.manual_seed(17)
        c = OFDMChannel()
        x = torch.ones(20000, 64, dtype=torch.complex64, device=DEVICE)
        y, state = c.sample(as_real(x), return_state=True)
        noise = as_complex(y)-state["h"]
        self.assertAlmostEqual(noise.real.var().item(), .05, delta=.001)
        empirical = state["h"].T @ state["h"].conj()/len(x)
        expected = c.frequency_covariance(DEVICE)
        self.assertLess((empirical-expected).norm().item()/expected.norm().item(), .04)
        response, _ = c.profile(DEVICE)
        self.assertGreater((response.imag.abs() > .1).float().mean().item(), .2)

    def test_shared_activity_mixture_and_covariance(self):
        torch.manual_seed(18)
        c = OFDMChannel(activity=.1)
        x = torch.zeros(30000, c.n, device=DEVICE)
        y, state = c.sample(x, return_state=True)
        active = state["activity"]
        energy = as_complex(y).abs().square()
        self.assertAlmostEqual(active.float().mean().item(), .1, delta=.01)
        self.assertAlmostEqual(energy[active].mean().item(), 1.1, delta=.02)
        self.assertAlmostEqual(energy[~active].mean().item(), .1, delta=.002)
        # Independent per-coordinate activity would give zero cross-energy covariance.
        cov = ((energy[:, 0]-energy[:, 0].mean())*(energy[:, 20]-energy[:, 20].mean())).mean()
        self.assertAlmostEqual(cov.item(), .09, delta=.02)
        self.assertAlmostEqual(c.covariance(x[:1], active=True)[0, 0, 0].real.item(), 1.1, places=5)

    def test_unitary_fft_and_cp_linear_equivalence(self):
        torch.manual_seed(19)
        c = OFDMChannel()
        x = complex_normal((5, 64), DEVICE)
        time = torch.fft.ifft(torch.fft.ifftshift(x, dim=-1), norm="ortho")
        torch.testing.assert_close(time.abs().square().sum(-1), x.abs().square().sum(-1))
        # Independent integer-delay test of the sufficient-CP frequency-domain model.
        delays = [0, 1, 3, 7, 12]
        taps = complex_normal((5, len(delays)), DEVICE)
        extended = torch.cat((time[:, -16:], time), -1)
        linear = sum(taps[:, i, None]*extended[:, 16-d:80-d] for i, d in enumerate(delays))
        frequencies = torch.arange(-32, 32, device=DEVICE)/64
        h = taps @ torch.exp(-2j*math.pi*torch.tensor(delays, device=DEVICE)[:, None]*frequencies)
        recovered = torch.fft.fftshift(torch.fft.fft(linear, norm="ortho"), dim=-1)
        torch.testing.assert_close(recovered, x*h, atol=2e-5, rtol=2e-5)
        with self.assertRaises(ValueError):
            OFDMChannel(delay_spread_s=10e-6)

    def test_covariance_orientation_and_pilot_estimator(self):
        c = OFDMChannel()
        receiver = PilotReceiver(c, 10., DEVICE)
        cov = c.frequency_covariance(DEVICE)
        pp = cov[list(c.pilot_indices)][:, list(c.pilot_indices)]+.1*torch.eye(8, device=DEVICE)
        torch.testing.assert_close(receiver.weights @ pp, cov[:, list(c.pilot_indices)], atol=2e-5, rtol=2e-5)
        labels = torch.randint(4, (20, 56), device=DEVICE)
        logits = receiver(c.sample(c.waveform(labels)))
        self.assertEqual(tuple(logits.shape), (20, 56, 4))
        self.assertTrue(torch.isfinite(logits).all())


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.official = load_sources(ROOT / "results/official_fast_sources_20261012")
        cls.wflow = load_official(ROOT / "results/official_sources_20261010/W-Flow")

    def test_all_family_finite_backward_and_exact_resume(self):
        for family, steps in (("gaussian", 1), ("mixture", 1), ("sinkhorn", 1), ("wgan", 1),
                              ("flow", 32), ("diffusion", 20), ("consistency", 1)):
            with self.subTest(family=family):
                channel = SSPACoverageChannel()
                cfg = FastConfig(family=family, parameters=4096, batch_size=128, updates=10, critic_steps=1)
                trainer = FastTrainer(cfg, DEVICE, self.official, channel, self.wflow)
                trainer.step()
                state = deepcopy(trainer.state())
                expected = trainer.step()
                resumed = FastTrainer(cfg, DEVICE, self.official, channel, self.wflow)
                resumed.restore(state)
                self.assertEqual(expected, resumed.step())
                x = torch.randn(3, 8, device=DEVICE, requires_grad=True)
                grad = torch.autograd.grad(FastImplant(resumed, steps)(x).square().mean(), x)[0]
                self.assertTrue(torch.isfinite(grad).all())

    def test_checkpoint_and_channel_identity(self):
        c = SSPACoverageChannel()
        cfg = FastConfig(parameters=4096, batch_size=128, updates=2)
        trainer = FastTrainer(cfg, DEVICE, self.official, c)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"state.pt"
            train_stage(trainer, path, 1, False)
            before = path.read_bytes()
            resumed = FastTrainer(cfg, DEVICE, self.official, c)
            train_stage(resumed, path, 1, True)
            self.assertEqual(before, path.read_bytes())
            wrong = FastTrainer(cfg, DEVICE, self.official, SSPACoverageChannel(input_law="normal"))
            with self.assertRaises(ValueError):
                wrong.restore(trainer.state())

    def test_receiver_gradient_resume_and_error_counts(self):
        c = OFDMChannel(activity=.1)
        points = Constellation(True).to(DEVICE)
        labels = torch.randint(4, (4, 56), device=DEVICE)
        c.sample(c.waveform(labels, points())).square().mean().backward()
        self.assertGreater(points.points.grad.norm().item(), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"receiver.pt"
            trained = train_receiver(c, c.sample, path, DEVICE, 42, 2, 8, 1, False, True)
            before = path.read_bytes()
            resumed = train_receiver(c, c.sample, path, DEVICE, 42, 2, 8, 1, True, True)
            self.assertEqual(before, path.read_bytes())
            a = evaluate_receiver(c, *trained[:2], DEVICE, 100, 32, 45, [10.])
            b = evaluate_receiver(c, *resumed[:2], DEVICE, 100, 32, 45, [10.])
            self.assertEqual(a, b)
            groups = a[0]["groups"]
            for key in ("blocks", "block_errors", "symbol_errors", "bit_errors"):
                self.assertEqual(groups["all"][key], groups["active"][key]+groups["inactive"][key])
            self.assertEqual(groups["all"]["bit_trials"], 11200)

    def test_frequency_receiver_checkpoint_and_input_gradient(self):
        from scripts.ofdm_frequency_training import train_receiver as frequency_train
        c = OFDMChannel(activity=.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"frequency.pt"
            receiver, constellation, _ = frequency_train(c, c.sample, path, DEVICE, 43, 2, 8, 1, False, True)
            x = c.waveform(torch.randint(4, (3, 56), device=DEVICE), constellation())
            gradient = torch.autograd.grad(receiver(c.sample(x)).square().mean(), constellation.points)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0.)
            before = path.read_bytes()
            frequency_train(c, c.sample, path, DEVICE, 43, 2, 8, 1, True, True)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
