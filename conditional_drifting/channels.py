from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

ChannelFn = Callable[[torch.Tensor, float, torch.device], torch.Tensor]


_TDL_STANDARD_PROFILES = {
    "tdl-d-lite": {
        "label": "TDL-D-lite",
        "source": "3GPP TR 38.901 v17.1.0, Table 7.7.2-4",
        "los_delay": 0.0,
        "los_power_db": -0.2,
        "los_k_factor_db": 13.3,
        "rayleigh_paths": (
            (0.0, -13.5),
            (0.035, -18.8),
            (0.612, -21.0),
            (1.363, -22.8),
            (1.405, -17.9),
            (1.804, -20.1),
            (2.596, -21.9),
            (1.775, -22.9),
            (4.042, -27.8),
            (7.937, -23.6),
            (9.424, -24.8),
            (9.708, -30.0),
            (12.525, -27.7),
        ),
    },
    "tdl-e-lite": {
        "label": "TDL-E-lite",
        "source": "3GPP TR 38.901 v17.1.0, Table 7.7.2-5",
        "los_delay": 0.0,
        "los_power_db": -0.03,
        "los_k_factor_db": 22.0,
        "rayleigh_paths": (
            (0.0, -22.03),
            (0.5133, -15.8),
            (0.5440, -18.1),
            (0.5630, -19.8),
            (0.5440, -22.9),
            (0.7112, -22.4),
            (1.9092, -18.6),
            (1.9293, -20.8),
            (1.9589, -22.6),
            (2.6426, -22.3),
            (3.7136, -25.6),
            (5.4524, -20.2),
            (12.0034, -29.8),
            (20.6519, -29.2),
        ),
    },
}

_TDL_PROFILE_ALIASES = {
    "tdl-d": "tdl-d-lite",
    "tdld": "tdl-d-lite",
    "tdl_d": "tdl-d-lite",
    "tdl_d_lite": "tdl-d-lite",
    "tdl-d-lite": "tdl-d-lite",
    "tdl-e": "tdl-e-lite",
    "tdle": "tdl-e-lite",
    "tdl_e": "tdl-e-lite",
    "tdl_e_lite": "tdl-e-lite",
    "tdl-e-lite": "tdl-e-lite",
    "custom": "custom",
    "legacy": "custom",
    "legacy-rician": "custom",
}


def _canonical_tdl_profile(profile: str | None) -> str:
    key = str(profile or "tdl-d-lite").strip().lower().replace(" ", "-")
    if key in _TDL_PROFILE_ALIASES:
        return _TDL_PROFILE_ALIASES[key]
    raise ValueError(f"Unsupported TDL profile: {profile!r}. Expected one of {sorted(_TDL_PROFILE_ALIASES)}.")


def _power_db_to_linear(power_db: float) -> float:
    return 10.0 ** (float(power_db) / 10.0)


def _linear_to_power_db(power: float) -> float:
    if float(power) <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(float(power))


def _tdl_shift(delay: float, delay_scale: float, symbols: int, circular: bool) -> int:
    shift = int(round(float(delay) * float(delay_scale)))
    if circular:
        return shift % max(1, int(symbols))
    return shift


def _build_tdl_tap_spec(
    *,
    symbols: int,
    profile: str,
    tap_power_db: tuple[float, ...] | None,
    los_k_factor_db: float | None,
    circular: bool,
    delay_scale: float,
) -> dict:
    canonical = _canonical_tdl_profile(profile)
    if symbols < 1:
        raise ValueError("TDL requires at least one complex symbol.")

    scatter_by_shift: dict[int, float] = {}
    los_by_shift: dict[int, float] = {}
    source_paths: list[dict[str, float | int | str]] = []

    if canonical == "custom":
        powers_db = tuple(tap_power_db if tap_power_db is not None else (0.0, -2.0, -6.0, -10.0))
        if not powers_db:
            raise ValueError("TDL requires at least one tap.")
        linear = [_power_db_to_linear(power) for power in powers_db]
        linear_sum = max(sum(linear), 1e-12)
        scatter_total = 1.0
        los_total = 0.0
        if los_k_factor_db is not None:
            k_linear = _power_db_to_linear(float(los_k_factor_db))
            los_total = k_linear / (k_linear + 1.0)
            scatter_total = 1.0 / (k_linear + 1.0)
        for tap_idx, power in enumerate(linear):
            shift = tap_idx % symbols if circular else tap_idx
            scatter_power = scatter_total * power / linear_sum
            scatter_by_shift[shift] = scatter_by_shift.get(shift, 0.0) + scatter_power
            source_paths.append(
                {
                    "kind": "rayleigh",
                    "delay": float(tap_idx),
                    "power_db": float(powers_db[tap_idx]),
                    "shift": int(shift),
                }
            )
        los_by_shift[0] = los_total
        source = "custom short-block TDL profile"
        label = "custom"
        k_factor = None if los_k_factor_db is None else float(los_k_factor_db)
    else:
        profile_cfg = _TDL_STANDARD_PROFILES[canonical]
        rayleigh_paths = profile_cfg["rayleigh_paths"]
        los_delay = float(profile_cfg["los_delay"])
        los_shift = _tdl_shift(los_delay, delay_scale, symbols, circular)
        los_power = _power_db_to_linear(float(profile_cfg["los_power_db"]))
        if los_k_factor_db is not None:
            same_delay_scatter = sum(
                _power_db_to_linear(float(power_db))
                for delay, power_db in rayleigh_paths
                if abs(float(delay) - los_delay) < 1e-12
            )
            if same_delay_scatter > 0.0:
                los_power = same_delay_scatter * _power_db_to_linear(float(los_k_factor_db))
        los_by_shift[los_shift] = los_by_shift.get(los_shift, 0.0) + los_power
        source_paths.append(
            {
                "kind": "los",
                "delay": los_delay,
                "power_db": _linear_to_power_db(los_power),
                "shift": int(los_shift),
            }
        )
        for delay, power_db in rayleigh_paths:
            shift = _tdl_shift(float(delay), delay_scale, symbols, circular)
            power = _power_db_to_linear(float(power_db))
            scatter_by_shift[shift] = scatter_by_shift.get(shift, 0.0) + power
            source_paths.append(
                {
                    "kind": "rayleigh",
                    "delay": float(delay),
                    "power_db": float(power_db),
                    "shift": int(shift),
                }
            )
        total_power = max(sum(scatter_by_shift.values()) + sum(los_by_shift.values()), 1e-12)
        scatter_by_shift = {shift: power / total_power for shift, power in scatter_by_shift.items()}
        los_by_shift = {shift: power / total_power for shift, power in los_by_shift.items()}
        source = str(profile_cfg["source"])
        label = str(profile_cfg["label"])
        k_factor = float(profile_cfg["los_k_factor_db"] if los_k_factor_db is None else los_k_factor_db)

    shifts = sorted(set(scatter_by_shift) | set(los_by_shift))
    scatter_powers = [float(scatter_by_shift.get(shift, 0.0)) for shift in shifts]
    los_powers = [float(los_by_shift.get(shift, 0.0)) for shift in shifts]
    total_powers = [scatter + los for scatter, los in zip(scatter_powers, los_powers)]
    return {
        "profile": canonical,
        "label": label,
        "source": source,
        "symbols": int(symbols),
        "circular": bool(circular),
        "delay_scale": float(delay_scale),
        "los_k_factor_db": k_factor,
        "tap_shifts": [int(shift) for shift in shifts],
        "scatter_powers": scatter_powers,
        "los_powers": los_powers,
        "total_powers": total_powers,
        "total_power_db": [_linear_to_power_db(power) for power in total_powers],
        "source_paths": source_paths,
    }


def tdl_effective_profile(
    *,
    symbols: int = 4,
    profile: str = "TDL-D-lite",
    tap_power_db: tuple[float, ...] | None = None,
    los_k_factor_db: float | None = None,
    circular: bool = True,
    delay_scale: float = 1.0,
) -> dict:
    """Return the discrete taps used by the compact TDL channel."""
    return _build_tdl_tap_spec(
        symbols=int(symbols),
        profile=profile,
        tap_power_db=tap_power_db,
        los_k_factor_db=los_k_factor_db,
        circular=circular,
        delay_scale=delay_scale,
    )


def awgn(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    noise = torch.normal(mean=0.0, std=float(noise_std), size=x.shape, device=device)
    return x + noise


def rayleigh_awgn(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    i, j = x.shape
    fading = (1.0 / math.sqrt(2.0)) * torch.sqrt(
        torch.randn((i, j), device=device).square() + torch.randn((i, j), device=device).square()
    )
    noise = float(noise_std) * torch.randn((i, j), device=device)
    return fading * x + noise


def modeflip_awgn(x: torch.Tensor, noise_std: float, device: torch.device, flip_prob: float = 0.5) -> torch.Tensor:
    """
    Bimodal synthetic channel for mode-collapse stress tests.

    For each sample, draw a Rademacher branch variable s in {+1, -1} and return
    y = s * x + n. For a fixed input x, this induces two separated modes.
    """
    x = x.to(device)
    if not (0.0 <= float(flip_prob) <= 1.0):
        raise ValueError("flip_prob must lie in [0, 1].")
    batch = x.shape[0]
    flips = torch.where(
        torch.rand((batch, 1), device=device) < float(flip_prob),
        torch.full((batch, 1), -1.0, device=device, dtype=x.dtype),
        torch.full((batch, 1), 1.0, device=device, dtype=x.dtype),
    )
    noise = float(noise_std) * torch.randn_like(x, device=device)
    return flips * x + noise


def sspa(x: torch.Tensor, noise_std: float, device: torch.device, p: float = 3.0, a0: float = 1.5, gain: float = 5.0) -> torch.Tensor:
    x = x.to(device)
    if x.shape[1] % 2 != 0:
        raise ValueError("SSPA expects an even channel dimension (I/Q pairs).")
    dim = x.shape[1] // 2
    x_2d = x.reshape(-1, 2)
    amplitude = torch.sum(x_2d.square(), dim=1).sqrt()
    amplitude_ratio = gain / (1.0 + (gain * amplitude / a0) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    x_amp = torch.mul(amplitude_ratio.reshape(-1, 1), x_2d).reshape(-1, 2 * dim)
    return x_amp + (float(noise_std) / math.sqrt(2.0)) * torch.randn_like(x_amp, device=device)


def tdl_rayleigh_awgn(
    x: torch.Tensor,
    noise_std: float,
    device: torch.device,
    *,
    profile: str = "TDL-D-lite",
    tap_power_db: tuple[float, ...] | None = None,
    los_k_factor_db: float | None = None,
    circular: bool = True,
    delay_scale: float = 1.0,
) -> torch.Tensor:
    """Short block tapped-delay-line fading channel with complex I/Q pairs.

    The tap gains are constant across one codeword and independent across
    samples. Circular convolution keeps the existing fixed-length channel API
    while still introducing inter-symbol memory. The default profile is a
    compact TDL-D profile derived from 3GPP TR 38.901, with normalized delays
    rounded to symbol-spaced lags in the short block.
    """
    x = x.to(device)
    if x.shape[1] % 2 != 0:
        raise ValueError("TDL expects an even channel dimension (I/Q pairs).")

    batch = x.shape[0]
    symbols = x.shape[1] // 2
    tap_spec = _build_tdl_tap_spec(
        symbols=symbols,
        profile=profile,
        tap_power_db=tap_power_db,
        los_k_factor_db=los_k_factor_db,
        circular=circular,
        delay_scale=delay_scale,
    )
    x_pairs = x.reshape(batch, symbols, 2)
    xr = x_pairs[..., 0]
    xi = x_pairs[..., 1]

    tap_shifts = [int(shift) for shift in tap_spec["tap_shifts"]]
    scatter_powers = torch.as_tensor(tap_spec["scatter_powers"], device=device, dtype=x.dtype)
    los_powers = torch.as_tensor(tap_spec["los_powers"], device=device, dtype=x.dtype)
    tap_scale = torch.sqrt(scatter_powers.clamp_min(0.0) / 2.0).reshape(1, -1)
    h_real = tap_scale * torch.randn((batch, len(tap_shifts)), device=device, dtype=x.dtype)
    h_imag = tap_scale * torch.randn((batch, len(tap_shifts)), device=device, dtype=x.dtype)
    h_real = h_real + torch.sqrt(los_powers.clamp_min(0.0)).reshape(1, -1)

    yr = torch.zeros_like(xr)
    yi = torch.zeros_like(xi)
    for local_idx, tap_shift in enumerate(tap_shifts):
        if circular:
            xr_shift = torch.roll(xr, shifts=tap_shift, dims=1)
            xi_shift = torch.roll(xi, shifts=tap_shift, dims=1)
        else:
            xr_shift = torch.zeros_like(xr)
            xi_shift = torch.zeros_like(xi)
            if tap_shift < symbols:
                xr_shift[:, tap_shift:] = xr[:, : symbols - tap_shift]
                xi_shift[:, tap_shift:] = xi[:, : symbols - tap_shift]
        hr = h_real[:, local_idx : local_idx + 1]
        hi = h_imag[:, local_idx : local_idx + 1]
        yr = yr + hr * xr_shift - hi * xi_shift
        yi = yi + hr * xi_shift + hi * xr_shift

    y = torch.stack((yr, yi), dim=-1).reshape_as(x)
    return y + float(noise_std) * torch.randn_like(y, device=device)


def optfib(
    x: torch.Tensor,
    noise_std: float,
    device: torch.device,
    *,
    L: float = 5000.0,
    gamma: float = 1.27,
    Kstep: int = 50,
    Pn_dBm: float = -21.3,
    use_noise_std: bool = False,
) -> torch.Tensor:
    x = x.to(device)
    if x.shape[1] % 2 != 0:
        raise ValueError("OptFib expects an even channel dimension (I/Q pairs).")

    if use_noise_std:
        sigma_n = float(noise_std) / math.sqrt(float(Kstep))
    else:
        sigma_n = float(np.sqrt((10 ** ((Pn_dBm - 30.0) / 10.0)) / float(Kstep) / 2.0))

    x2 = x.reshape(-1, 2)
    xr = x2[:, 0:1]
    xi = x2[:, 1:2]
    for _ in range(int(Kstep)):
        phase = gamma * (xr.square() + xi.square()) * L / float(Kstep)
        xr_rot = xr * torch.cos(phase) - xi * torch.sin(phase)
        xi_rot = xi * torch.cos(phase) + xr * torch.sin(phase)
        xr = xr_rot + sigma_n * torch.randn_like(xr, device=device)
        xi = xi_rot + sigma_n * torch.randn_like(xi, device=device)
    return torch.cat((xr, xi), dim=1).reshape_as(x)


def channel_registry(
    optfib_params: dict | None = None,
    modeflip_params: dict | None = None,
    tdl_params: dict | None = None,
) -> dict[str, ChannelFn]:
    params = dict(optfib_params or {})
    modeflip_cfg = dict(modeflip_params or {})
    tdl_cfg = dict(tdl_params or {})

    def _optfib(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
        return optfib(x, noise_std, device, **params)

    def _modeflip(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
        return modeflip_awgn(x, noise_std, device, **modeflip_cfg)

    def _tdl(x: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
        return tdl_rayleigh_awgn(x, noise_std, device, **tdl_cfg)

    return {
        "AWGN": awgn,
        "Rayleigh": rayleigh_awgn,
        "ModeFlip": _modeflip,
        "SSPA": sspa,
        "TDL": _tdl,
        "OptFib": _optfib,
    }
