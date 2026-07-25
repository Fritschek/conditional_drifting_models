from __future__ import annotations

from importlib import import_module

__all__ = [
    "DiffusionConfig",
    "GANConfig",
    "PaperWGANConfig",
    "evaluate_diffusion_model",
    "evaluate_gan_model",
    "evaluate_paper_wgan",
    "sample_ddim",
    "sample_ddpm",
    "train_paper_wgan",
    "train_conditional_diffusion",
    "train_conditional_gan",
]

_MODULE_EXPORTS = {
    "DiffusionConfig": ("conditional_drifting.baselines.diffusion", "DiffusionConfig"),
    "GANConfig": ("conditional_drifting.baselines.gan", "GANConfig"),
    "PaperWGANConfig": ("conditional_drifting.baselines.paper_wgan", "PaperWGANConfig"),
    "evaluate_diffusion_model": ("conditional_drifting.baselines.diffusion", "evaluate_diffusion_model"),
    "evaluate_gan_model": ("conditional_drifting.baselines.gan", "evaluate_gan_model"),
    "evaluate_paper_wgan": ("conditional_drifting.baselines.paper_wgan", "evaluate_paper_wgan"),
    "sample_ddim": ("conditional_drifting.baselines.diffusion", "sample_ddim"),
    "sample_ddpm": ("conditional_drifting.baselines.diffusion", "sample_ddpm"),
    "train_paper_wgan": ("conditional_drifting.baselines.paper_wgan", "train_paper_wgan"),
    "train_conditional_diffusion": ("conditional_drifting.baselines.diffusion", "train_conditional_diffusion"),
    "train_conditional_gan": ("conditional_drifting.baselines.gan", "train_conditional_gan"),
}


def __getattr__(name: str):
    if name in _MODULE_EXPORTS:
        module_name, attr_name = _MODULE_EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
