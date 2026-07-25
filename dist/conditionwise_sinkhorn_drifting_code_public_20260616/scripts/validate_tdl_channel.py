from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import tdl_effective_profile, tdl_rayleigh_awgn
from conditional_drifting.training import set_seed


def parse_float_csv(text: str | None) -> tuple[float, ...] | None:
    if text is None or not text.strip():
        return None
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the compact short-block TDL channel profile.")
    parser.add_argument("--profile", type=str, default="TDL-D-lite")
    parser.add_argument("--symbols", type=int, default=4)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--tap-power-db", type=str, default=None)
    parser.add_argument("--los-k-factor-db", type=float, default=None)
    parser.add_argument("--delay-scale", type=float, default=1.0)
    parser.add_argument("--no-circular", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    circular = not args.no_circular
    tap_power_db = parse_float_csv(args.tap_power_db)

    profile = tdl_effective_profile(
        symbols=args.symbols,
        profile=args.profile,
        tap_power_db=tap_power_db,
        los_k_factor_db=args.los_k_factor_db,
        circular=circular,
        delay_scale=args.delay_scale,
    )

    samples = int(args.samples)
    x = torch.randn((samples, 2 * int(args.symbols)), device=device)
    y_signal = tdl_rayleigh_awgn(
        x,
        0.0,
        device,
        profile=args.profile,
        tap_power_db=tap_power_db,
        los_k_factor_db=args.los_k_factor_db,
        circular=circular,
        delay_scale=args.delay_scale,
    )
    y_noisy = tdl_rayleigh_awgn(
        x,
        float(args.noise_std),
        device,
        profile=args.profile,
        tap_power_db=tap_power_db,
        los_k_factor_db=args.los_k_factor_db,
        circular=circular,
        delay_scale=args.delay_scale,
    )

    impulse = torch.zeros((samples, 2 * int(args.symbols)), device=device)
    impulse[:, 0] = 1.0
    impulse_response = tdl_rayleigh_awgn(
        impulse,
        0.0,
        device,
        profile=args.profile,
        tap_power_db=tap_power_db,
        los_k_factor_db=args.los_k_factor_db,
        circular=circular,
        delay_scale=args.delay_scale,
    ).reshape(samples, int(args.symbols), 2)
    empirical_impulse_power = impulse_response.square().sum(dim=2).mean(dim=0).detach().cpu()

    expected_impulse_power = torch.zeros(int(args.symbols), dtype=torch.float64)
    for shift, total_power in zip(profile["tap_shifts"], profile["total_powers"]):
        if 0 <= int(shift) < int(args.symbols):
            expected_impulse_power[int(shift)] += float(total_power)

    payload = {
        "seed": int(args.seed),
        "device": str(device),
        "samples": samples,
        "noise_std": float(args.noise_std),
        "effective_profile": profile,
        "input_power_per_real_dim": float(x.square().mean().detach().cpu()),
        "signal_output_power_per_real_dim": float(y_signal.square().mean().detach().cpu()),
        "noisy_output_power_per_real_dim": float(y_noisy.square().mean().detach().cpu()),
        "empirical_impulse_power_by_symbol": [float(v) for v in empirical_impulse_power],
        "expected_impulse_power_by_symbol": [float(v) for v in expected_impulse_power],
        "max_abs_impulse_power_error": float(torch.max(torch.abs(empirical_impulse_power.double() - expected_impulse_power))),
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
