"""Versioned physical contracts for the post-review application experiments."""

from dataclasses import asdict, dataclass
import math

import torch

from .channels import sspa


def real_noise_variance(energy_per_bit, ebno_db):
    """Complex-baseband convention: each real component has variance N0/2."""
    if energy_per_bit <= 0 or not math.isfinite(energy_per_bit + ebno_db):
        raise ValueError("Finite positive bit energy and finite Eb/N0 required")
    return energy_per_bit / (2 * 10 ** (ebno_db / 10))


@dataclass(frozen=True)
class SSPACoverageChannel:
    version: str = "sspa_input_ebno_v1"
    n: int = 8
    messages: int = 64
    # Same physical variance as historical nominal 8 dB; now explicitly labeled.
    ebno_db: float = 8 + 10 * math.log10(2)
    input_law: str = "broad"

    def __post_init__(self):
        if self.n != 8 or self.messages != 64 or self.input_law not in ("normal", "broad"):
            raise ValueError("This version fixes the eight-real-coordinate M64 experiment")

    @property
    def variance(self):
        return real_noise_variance(self.n / math.log2(self.messages), self.ebno_db)

    def description(self):
        return dict(asdict(self), real_noise_variance=self.variance,
                    ebno_reference="input codeword energy / information bits; real variance=N0/2",
                    input_energy=self.n, expected_query_energy=self.n if self.input_law == "normal" else 146/3)

    def inputs(self, count, device, generator):
        x = torch.randn(count, self.n, device=device, generator=generator)
        if self.input_law == "broad":
            u = torch.rand(count, 2, device=device, generator=generator)
            radial = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            radial = radial * u[:, 1:] * math.sqrt(self.messages * self.n)
            x = torch.where(u[:, :1] < .25, radial, x)
        return x

    def mean(self, x):
        return sspa(x, 0., x.device)

    def sample(self, x, ebno_db=None, generator=None):
        variance = self.variance if ebno_db is None else real_noise_variance(
            self.n / math.log2(self.messages), ebno_db)
        return self.mean(x) + math.sqrt(variance) * torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator)


# Numerical profile from Sionna v1.2.2, TR 38.901 Table 7.7.2-3.
# https://github.com/NVlabs/sionna/blob/v1.2.2/src/sionna/phy/channel/tr38901/models/TDL-C.json
TDLC_DELAYS = (0., .2099, .2219, .2329, .2176, .6366, .6448, .6560, .6584,
               .7935, .8213, .9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589,
               4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523)
TDLC_POWERS_DB = (-4.4, -1.2, -3.5, -5.2, -2.5, 0., -2.2, -3.9, -7.4,
                   -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9,
                   -13.9, -15.8, -17.1, -16., -15.7, -21.6, -22.8)


def as_complex(x):
    return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())


def as_real(x):
    return torch.view_as_real(x).flatten(-2)


def complex_normal(shape, device, generator=None, dtype=torch.float32):
    return torch.view_as_complex(torch.randn(*shape, 2, device=device, dtype=dtype,
                                            generator=generator)) / math.sqrt(2)


@dataclass(frozen=True)
class OFDMChannel:
    version: str = "ofdm64_gaussian_tdlc_v1"
    subcarriers: int = 64
    spacing_hz: float = 30000.
    cp_samples: int = 16
    delay_spread_s: float = 300e-9
    esno_db: float = 10.
    activity: float = 0.
    interference_power: float = 1.
    input_law: str = "mixed"

    def __post_init__(self):
        if self.subcarriers != 64 or self.cp_samples < 0 or self.spacing_hz <= 0 or self.delay_spread_s <= 0:
            raise ValueError("Invalid OFDM configuration")
        if not 0 <= self.activity <= 1 or self.interference_power < 0 or self.input_law not in ("mixed", "qpsk"):
            raise ValueError("Invalid input/interference law")
        if max(TDLC_DELAYS) * self.delay_spread_s > self.cp_samples / (self.subcarriers*self.spacing_hz):
            raise ValueError("Delay profile exceeds cyclic prefix")

    @property
    def n(self):
        return 2 * self.subcarriers

    @property
    def pilot_indices(self):
        return tuple(range(0, self.subcarriers, 8))

    @property
    def data_indices(self):
        return tuple(i for i in range(self.subcarriers) if i not in self.pilot_indices)

    def description(self):
        return dict(asdict(self), n=self.n, pilots=self.pilot_indices, data=self.data_indices,
                    pilot_value=[1., 0.], frequency_order="fftshift: -32..31",
                    real_noise_variance=10**(-self.esno_db/10)/2,
                    esno_reference="unit average energy per occupied subcarrier / complex thermal N0",
                    profile_delays=TDLC_DELAYS, profile_powers_db=TDLC_POWERS_DB,
                    fading="independent proper Gaussian taps; shared within block; no receiver CSI",
                    interference="independent proper Gaussian coordinates; one hidden Bernoulli per block",
                    implementation="frequency response at fractional delays; sufficient-CP block-static model, not Sionna SoS",
                    guards="all 64 carriers occupied, including DC; no RF guard-band model")

    def profile(self, device, dtype=torch.float32):
        delays = torch.tensor(TDLC_DELAYS, device=device, dtype=dtype) * self.delay_spread_s
        powers = 10**(torch.tensor(TDLC_POWERS_DB, device=device, dtype=dtype)/10)
        powers = powers / powers.sum()
        frequencies = (torch.arange(self.subcarriers, device=device, dtype=dtype)-self.subcarriers//2)*self.spacing_hz
        response = torch.exp(-2j*math.pi*frequencies[:, None]*delays[None, :])
        return response, powers

    def frequency_covariance(self, device, dtype=torch.float32):
        response, powers = self.profile(device, dtype)
        return (response * powers) @ response.mH

    def waveform(self, labels, constellation=None):
        if labels.shape[-1] != len(self.data_indices):
            raise ValueError("One QPSK label per data carrier required")
        if constellation is None:
            constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=labels.device)/math.sqrt(2)
        x = torch.ones(*labels.shape[:-1], self.subcarriers, device=labels.device, dtype=constellation.dtype)
        x[..., list(self.data_indices)] = constellation[labels]
        return as_real(x)

    def inputs(self, count, device, generator):
        labels = torch.randint(4, (count, len(self.data_indices)), device=device, generator=generator)
        x = self.waveform(labels)
        if self.input_law == "mixed":
            gaussian = as_real(complex_normal((count, self.subcarriers), device, generator))
            # Pilot values remain fixed in both mixture components.
            gaussian.reshape(count, self.subcarriers, 2)[:, list(self.pilot_indices)] = x.reshape(count, self.subcarriers, 2)[:, list(self.pilot_indices)]
            mask = torch.rand(count, 1, device=device, generator=generator) < .5
            x = torch.where(mask, gaussian, x)
        return x

    def sample(self, x, esno_db=None, generator=None, return_state=False):
        xc = as_complex(x)
        response, powers = self.profile(x.device, x.dtype)
        taps = complex_normal((len(x), len(powers)), x.device, generator, x.dtype)*powers.sqrt()
        h = taps @ response.T
        activity = torch.rand(len(x), 1, device=x.device, generator=generator) < self.activity
        n0 = 10**(-(self.esno_db if esno_db is None else esno_db)/10)
        noise = complex_normal(xc.shape, x.device, generator, x.dtype)*math.sqrt(n0)
        interference = complex_normal(xc.shape, x.device, generator, x.dtype)*math.sqrt(self.interference_power)
        y = as_real(h*xc + noise + activity*interference)
        return (y, dict(activity=activity[:, 0], h=h)) if return_state else y

    def covariance(self, x, active=False):
        xc = as_complex(x)
        covariance = xc.unsqueeze(-1)*self.frequency_covariance(x.device, x.dtype)*xc.conj().unsqueeze(-2)
        variance = 10**(-self.esno_db/10) + (self.interference_power if active else 0.)
        return covariance + variance*torch.eye(self.subcarriers, device=x.device)
