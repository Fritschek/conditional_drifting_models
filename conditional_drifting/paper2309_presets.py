from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperChannelPreset:
    n: int
    ebn0_db: float
    rate: float
    diffusion_hidden_dim: int
    wgan_hidden_dim: int
    diffusion_epochs: int = 30
    drifting_epochs: int = 30
    wgan_epochs: int = 30
    dataset_size: int = 10_000_000
    batch_size: int = 5_000
    eval_size: int = 10_000_000
    swd_projections: int = 128
    num_steps: int = 100
    diffusion_learning_rate_schedule: tuple[tuple[int, float], ...] | None = None


PAPER2309_PRESETS: dict[str, PaperChannelPreset] = {
    "AWGN": PaperChannelPreset(
        n=7,
        ebn0_db=5.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=110,
        wgan_hidden_dim=128,
        diffusion_learning_rate_schedule=((10, 1e-3), (20, 1e-4)),
    ),
    "Rayleigh": PaperChannelPreset(
        n=7,
        ebn0_db=12.0,
        rate=4.0 / 7.0,
        diffusion_hidden_dim=128,
        wgan_hidden_dim=256,
        diffusion_learning_rate_schedule=((10, 1e-3), (20, 1e-4)),
    ),
    "SSPA": PaperChannelPreset(
        n=8,
        ebn0_db=8.0,
        rate=6.0 / 8.0,
        diffusion_hidden_dim=110,
        wgan_hidden_dim=256,
        diffusion_epochs=160,
        drifting_epochs=160,
        wgan_epochs=160,
        batch_size=4096,
    ),
    "TDL": PaperChannelPreset(
        n=8,
        ebn0_db=10.0,
        rate=4.0 / 8.0,
        diffusion_hidden_dim=128,
        wgan_hidden_dim=256,
        diffusion_epochs=60,
        drifting_epochs=60,
        wgan_epochs=60,
        dataset_size=120_000,
        batch_size=512,
        eval_size=100_000,
        swd_projections=128,
    ),
}


def ebno_to_noise(ebn0_db: float, rate: float) -> float:
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / math.sqrt(2.0 * rate * ebn0)


def resolve_paper_diffusion_lr_schedule(
    preset: PaperChannelPreset,
    diffusion_epochs: int,
) -> tuple[tuple[int, float], ...] | None:
    schedule = preset.diffusion_learning_rate_schedule
    if schedule is None:
        return None
    if sum(stage_epochs for stage_epochs, _ in schedule) == diffusion_epochs:
        return schedule
    return None
