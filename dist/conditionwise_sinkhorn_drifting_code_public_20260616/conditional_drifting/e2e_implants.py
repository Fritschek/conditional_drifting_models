from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .baselines.diffusion import ConditionalDiffusionMLP, sample_ddim, sample_ddpm
from .baselines.paper_wgan import PaperWGANGenerator
from .channels import channel_registry, optfib
from .model import ConditionalDriftingGenerator
from .training import build_condition_context, build_condition_features, condition_context_dim, condition_feature_dim


def _flatten_feature_vectors(encoded_data: torch.Tensor, feature_dim: int) -> tuple[torch.Tensor, torch.Size]:
    if encoded_data.ndim < 2 or encoded_data.shape[-1] != feature_dim:
        raise ValueError(
            f"Channel implant expected encoded data with last dimension {feature_dim}, "
            f"but got shape {tuple(encoded_data.shape)}."
        )
    original_shape = encoded_data.shape
    flat = encoded_data.reshape(-1, feature_dim)
    return flat, original_shape


def _restore_feature_vectors(flat: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    return flat.reshape(original_shape)


class AnalyticChannelImplant:
    """Analytic memoryless channel implant using the repo channel functions."""

    respects_ebno = True

    def __init__(self, channel_name: str = "AWGN", *, optfib_params: dict[str, Any] | None = None):
        registry = channel_registry(optfib_params)
        if channel_name not in registry:
            raise ValueError(f"Unsupported analytic channel: {channel_name}")
        self.channel_name = str(channel_name)
        self.name = f"analytic_{self.channel_name.lower()}"
        self._channel_fn = registry[self.channel_name]

    def __call__(
        self,
        encoded_data: torch.Tensor,
        *,
        ebno_db: float,
        rate: float,
        device: torch.device | None = None,
        decoder_training: bool = False,
        ebno_range: tuple[float, float] = (-3.5, 0.0),
    ) -> torch.Tensor:
        device = device or encoded_data.device
        encoded_data = encoded_data.to(device)
        original_shape = encoded_data.shape
        channel_input = encoded_data.reshape(encoded_data.shape[0], -1) if encoded_data.ndim > 2 else encoded_data
        if decoder_training and self.channel_name == "AWGN":
            low = float(ebno_db + ebno_range[0])
            high = float(ebno_db + ebno_range[1])
            ebno_db_tensor = torch.empty_like(channel_input, device=device).uniform_(low, high)
            ebno_linear = torch.pow(10.0, ebno_db_tensor / 10.0)
        else:
            ebno_linear = torch.tensor(10.0 ** (float(ebno_db) / 10.0), device=device, dtype=channel_input.dtype)

        signal_power = torch.mean(channel_input.square())
        noise_power = signal_power / (2.0 * float(rate) * ebno_linear)
        if torch.is_tensor(noise_power):
            noise_std = float(torch.sqrt(noise_power.mean()).detach().cpu().item())
        else:
            noise_std = math.sqrt(float(noise_power))
        channel_output = self._channel_fn(channel_input, noise_std, device)
        return channel_output.reshape(original_shape)


class AnalyticAWGNImplant(AnalyticChannelImplant):
    """Exact AWGN implant matching the symbolic benchmark interface."""

    def __init__(self):
        super().__init__("AWGN")
        self.name = "analytic_awgn"


class _PerSymbolImplant:
    respects_ebno = False
    input_dim: int

    def __call__(self, encoded_data: torch.Tensor, **_: Any) -> torch.Tensor:
        flat, original_shape = _flatten_feature_vectors(encoded_data, self.input_dim)
        flat = flat.to(next(self._module.parameters()).device if isinstance(self._module, nn.Module) else encoded_data.device)
        out = self._sample_flat(flat)
        return _restore_feature_vectors(out, original_shape).to(encoded_data.device)

    def _sample_flat(self, flat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DriftingChannelImplant(_PerSymbolImplant):
    def __init__(
        self,
        model: ConditionalDriftingGenerator,
        *,
        is_residual: bool,
        condition_input_scale: float = 1.0,
        condition_context_mode: str = "input",
        condition_feature_mode: str = "raw",
        optfib_gamma: float = 1.27,
        optfib_length: float = 5000.0,
        physics_base_mode: str = "identity",
        physics_base_optfib_kstep: int = 20,
        physics_base_optfib_pn_dbm: float = -21.3,
    ):
        self._module = model.eval()
        for param in self._module.parameters():
            param.requires_grad_(False)
        self.input_dim = model.net[-1].out_features
        self.is_residual = bool(is_residual)
        self.condition_input_scale = float(condition_input_scale)
        self.condition_context_mode = str(condition_context_mode or "input")
        self.condition_feature_mode = str(condition_feature_mode or "raw")
        self.optfib_gamma = float(optfib_gamma)
        self.optfib_length = float(optfib_length)
        self.physics_base_mode = str(physics_base_mode or "identity")
        self.physics_base_optfib_kstep = int(physics_base_optfib_kstep)
        self.physics_base_optfib_pn_dbm = float(physics_base_optfib_pn_dbm)
        self.name = self._resolve_name()

    def _resolve_name(self) -> str:
        if self._uses_physics_base():
            return f"drifting_{self.physics_base_mode}_base_residual"
        return "drifting_residual" if self.is_residual else "drifting_direct"

    def _uses_physics_base(self) -> bool:
        return str(self.physics_base_mode or "identity").lower() != "identity"

    def _sample_physics_base(self, flat: torch.Tensor) -> torch.Tensor:
        mode = str(self.physics_base_mode or "identity").lower()
        if mode == "identity":
            return flat
        if mode == "optfib":
            return optfib(
                flat,
                0.0,
                flat.device,
                Kstep=int(self.physics_base_optfib_kstep),
                Pn_dBm=float(self.physics_base_optfib_pn_dbm),
            )
        if mode == "optfib_noiseless":
            return optfib(
                flat,
                0.0,
                flat.device,
                Kstep=max(1, int(self.physics_base_optfib_kstep)),
                use_noise_std=True,
            )
        raise ValueError(f"Unsupported physics_base_mode={self.physics_base_mode!r}.")

    def _sample_flat(self, flat: torch.Tensor) -> torch.Tensor:
        base = self._sample_physics_base(flat) if self._uses_physics_base() else None
        context = build_condition_context(
            flat,
            base_condition=base,
            context_mode=self.condition_context_mode,
        )
        generated = self._module(
            build_condition_features(
                context,
                input_scale=self.condition_input_scale,
                feature_mode=self.condition_feature_mode,
                optfib_gamma=self.optfib_gamma,
                optfib_length=self.optfib_length,
            )
        )
        if base is not None:
            return base + generated
        if self.is_residual:
            return flat + generated
        return generated


class PaperWGANChannelImplant(_PerSymbolImplant):
    def __init__(self, generator: PaperWGANGenerator):
        self._module = generator.eval()
        for param in self._module.parameters():
            param.requires_grad_(False)
        self.input_dim = generator.net[-1].out_features
        self.name = "paper_wgan"

    def _sample_flat(self, flat: torch.Tensor) -> torch.Tensor:
        return self._module(flat)


class DiffusionChannelImplant(_PerSymbolImplant):
    def __init__(self, model: nn.Module, *, sampler: str = "ddim", ddim_steps: int | None = 100):
        self._module = model.eval()
        for param in self._module.parameters():
            param.requires_grad_(False)
        self.input_dim = getattr(model, "n")
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        self.name = "ddpm" if sampler == "ddpm" else f"ddim{ddim_steps or 'full'}"

    def _sample_flat(self, flat: torch.Tensor) -> torch.Tensor:
        state = getattr(self._module, "_diffusion_state")
        if self.sampler == "ddpm":
            target = sample_ddpm(self._module, flat, state)
        elif self.sampler == "ddim":
            target = sample_ddim(self._module, flat, state, self.ddim_steps)
        else:
            raise ValueError(f"Unsupported diffusion sampler: {self.sampler}")
        if state.get("is_residual", True):
            return flat + target
        return target


def save_implant_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    family: str,
    metadata: dict[str, Any] | None = None,
    is_residual: bool | None = None,
) -> None:
    path = Path(path)
    payload: dict[str, Any] = {
        "family": family,
        "metadata": dict(metadata or {}),
        "model_state": model.state_dict(),
    }

    if family == "drifting":
        if not isinstance(model, ConditionalDriftingGenerator):
            raise TypeError("Drifting checkpoints require a ConditionalDriftingGenerator.")
        payload["config"] = {
            "condition_dim": model.net[0].in_features - model.latent_dim,
            "output_dim": model.net[-1].out_features,
            "latent_dim": model.latent_dim,
            "latent_input_scale": float(getattr(model, "latent_input_scale", 1.0)),
            "hidden_dim": model.net[0].out_features,
            "is_residual": bool(is_residual),
            "condition_input_scale": float((metadata or {}).get("condition_input_scale", 1.0)),
            "condition_context_mode": str((metadata or {}).get("condition_context_mode", "input")),
            "condition_feature_mode": str((metadata or {}).get("condition_feature_mode", "raw")),
            "optfib_gamma": float((metadata or {}).get("optfib_gamma", 1.27)),
            "optfib_length": float((metadata or {}).get("optfib_length", 5000.0)),
            "physics_base_mode": str((metadata or {}).get("physics_base_mode", "identity")),
            "physics_base_optfib_kstep": int((metadata or {}).get("physics_base_optfib_kstep", 20)),
            "physics_base_optfib_pn_dbm": float((metadata or {}).get("physics_base_optfib_pn_dbm", -21.3)),
        }
    elif family == "paper_wgan":
        if not isinstance(model, PaperWGANGenerator):
            raise TypeError("WGAN checkpoints require a PaperWGANGenerator.")
        payload["config"] = {
            "n": model.net[0].in_features // 2,
            "hidden_dim": model.net[0].out_features,
        }
    elif family == "diffusion":
        if not isinstance(model, ConditionalDiffusionMLP):
            raise TypeError("Diffusion checkpoints require a ConditionalDiffusionMLP.")
        payload["config"] = {
            "n": model.n,
            "hidden_dim": model.layer1.out_features,
            "num_steps": model.num_steps,
        }
        payload["diffusion_state"] = getattr(model, "_diffusion_state")
    else:
        raise ValueError(f"Unsupported implant family: {family}")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_implant_from_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    diffusion_sampler: str = "ddim",
    ddim_steps: int | None = 100,
):
    payload = torch.load(Path(path), map_location=device)
    if "family" not in payload:
        if "model_state_dict" in payload and "config" in payload:
            cfg = payload["config"]
            feature_mode = str(cfg.get("condition_feature_mode", "raw"))
            context_mode = str(cfg.get("condition_context_mode", "input"))
            model = ConditionalDriftingGenerator(
                condition_dim=condition_feature_dim(
                    condition_context_dim(int(cfg["n"]), context_mode),
                    feature_mode,
                ),
                output_dim=int(cfg["n"]),
                latent_dim=int(cfg["latent_dim"]),
                hidden_dim=int(cfg["hidden_dim"]),
                latent_input_scale=float(cfg.get("latent_input_scale", 1.0)),
            ).to(device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            return DriftingChannelImplant(
                model,
                is_residual=bool(cfg.get("is_residual", False)),
                condition_input_scale=float(cfg.get("condition_input_scale", 1.0)),
                condition_context_mode=context_mode,
                condition_feature_mode=feature_mode,
                optfib_gamma=float(cfg.get("optfib_gamma", 1.27)),
                optfib_length=float(cfg.get("optfib_length", 5000.0)),
                physics_base_mode=str(cfg.get("physics_base_mode", "identity")),
                physics_base_optfib_kstep=int(cfg.get("physics_base_optfib_kstep", 20)),
                physics_base_optfib_pn_dbm=float(cfg.get("physics_base_optfib_pn_dbm", -21.3)),
            )
        raise KeyError("Checkpoint payload does not contain a supported implant format.")
    family = payload["family"]

    if family == "drifting":
        cfg = payload["config"]
        feature_mode = str(cfg.get("condition_feature_mode", "raw"))
        context_mode = str(cfg.get("condition_context_mode", "input"))
        model = ConditionalDriftingGenerator(
            condition_dim=int(
                cfg.get(
                    "condition_dim",
                    condition_feature_dim(
                        condition_context_dim(int(cfg["output_dim"]), context_mode),
                        feature_mode,
                    ),
                )
            ),
            output_dim=int(cfg["output_dim"]),
            latent_dim=int(cfg["latent_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            latent_input_scale=float(cfg.get("latent_input_scale", 1.0)),
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        return DriftingChannelImplant(
            model,
            is_residual=bool(cfg["is_residual"]),
            condition_input_scale=float(cfg.get("condition_input_scale", 1.0)),
            condition_context_mode=context_mode,
            condition_feature_mode=feature_mode,
            optfib_gamma=float(cfg.get("optfib_gamma", 1.27)),
            optfib_length=float(cfg.get("optfib_length", 5000.0)),
            physics_base_mode=str(cfg.get("physics_base_mode", "identity")),
            physics_base_optfib_kstep=int(cfg.get("physics_base_optfib_kstep", 20)),
            physics_base_optfib_pn_dbm=float(cfg.get("physics_base_optfib_pn_dbm", -21.3)),
        )

    if family == "paper_wgan":
        cfg = payload["config"]
        model = PaperWGANGenerator(
            n=int(cfg["n"]),
            hidden_dim=int(cfg["hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        return PaperWGANChannelImplant(model)

    if family == "diffusion":
        cfg = payload["config"]
        model = ConditionalDiffusionMLP(
            n=int(cfg["n"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_steps=int(cfg["num_steps"]),
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model._diffusion_state = payload["diffusion_state"]
        model.eval()
        return DiffusionChannelImplant(model, sampler=diffusion_sampler, ddim_steps=ddim_steps)

    raise ValueError(f"Unsupported implant family in checkpoint: {family}")
