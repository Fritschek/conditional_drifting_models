"""Capacity-controlled channel adapters for official fast generative methods."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..channels import channel_registry, tdl_effective_profile
from ..paper2309_presets import PAPER2309_PRESETS, ebno_to_noise
from ..sspa_trajectory import rng_state, restore_rng
from ..training import set_seed

FAMILIES = ("gaussian", "diffusion", "flow", "consistency", "sinkhorn", "wgan")


@dataclass(frozen=True)
class FastConfig:
    channel: str = "SSPA"
    seed: int = 9101
    family: str = "flow"
    parameters: int = 60000
    batch_size: int = 4096
    updates: int = 20000
    train_seconds: float = 0.
    learning_rate: float = .0002
    ema_decay: float = .999
    inference_weights: str = "ema"
    samples_per_input: int = 64
    sinkhorn_iterations: int = 10
    sinkhorn_r: float = .05
    diffusion_steps: int = 1000
    consistency_sigma_min: float = .002
    consistency_sigma_max: float = 80.
    consistency_sigma_data: float = .5
    consistency_end_scales: int = 150
    consistency_two_step_index: int = 22
    gaussian_covariance: str = "diagonal"
    critic_steps: int = 5
    critic_clip: float = .01

    def validate(self):
        if self.channel not in PAPER2309_PRESETS or self.family not in FAMILIES:
            raise ValueError("Unknown channel or family")
        if min(self.parameters, self.batch_size, self.updates, self.critic_steps) <= 0:
            raise ValueError("Positive capacity, batch, update and critic counts required")
        if self.family == "sinkhorn" and (self.samples_per_input < 2 or self.batch_size % self.samples_per_input):
            raise ValueError("Sinkhorn batch must divide into clouds with K >= 2")
        if self.inference_weights not in ("raw", "ema") or self.gaussian_covariance not in ("diagonal", "full"):
            raise ValueError("Invalid inference weights or Gaussian covariance")
        if self.learning_rate <= 0 or self.train_seconds < 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid optimizer/budget configuration")
        if not 0 < self.consistency_sigma_min < self.consistency_sigma_max or self.consistency_end_scales < 2:
            raise ValueError("Invalid consistency schedule")
        if not 0 < self.consistency_two_step_index < 39:
            raise ValueError("Two-step intermediate index must be inside the 40-point schedule")
        if min(self.sinkhorn_iterations, self.sinkhorn_r, self.consistency_sigma_data, self.critic_clip) <= 0 or self.diffusion_steps < 2:
            raise ValueError("Invalid solver or noise configuration")


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def width_for(inputs, outputs, target):
    width = min(range(8, 2049), key=lambda h: abs(h * h + h * (inputs + outputs + 2) + outputs - target))
    count = width * width + width * (inputs + outputs + 2) + outputs
    if abs(count / target - 1) > .1:
        raise ValueError("Requested capacity cannot be matched within 10%")
    return width


def mlp(inputs, outputs, target, activation=nn.SiLU):
    width = width_for(inputs, outputs, target)
    return nn.Sequential(nn.Linear(inputs, width), activation(), nn.Linear(width, width),
                         activation(), nn.Linear(width, outputs))


class ChannelNetwork(nn.Module):
    def __init__(self, n, target, timed, embedding, latent_dim=None, activation=nn.SiLU):
        super().__init__()
        self.timed, self.embedding = timed, embedding
        self.latent_dim = n if latent_dim is None else latent_dim
        self.net = mlp(n + self.latent_dim + (32 if timed else 0), n, target, activation)

    def forward(self, state, t=None, condition=None):
        if condition is None:
            raise ValueError("The transmitted input must be supplied separately")
        parts = [state, condition]
        if self.timed:
            parts.append(self.embedding(t, 32).to(state))
        return self.net(torch.cat(parts, -1))


class GaussianNetwork(nn.Module):
    def __init__(self, n, target, full=False):
        super().__init__()
        self.n, self.full = n, full
        self.net = mlp(n, n + (n * (n + 1) // 2 if full else n), target)
        self.register_buffer("indices", torch.tril_indices(n, n), persistent=False)

    def distribution(self, x):
        values = self.net(x)
        mean, raw = values[:, :self.n], values[:, self.n:]
        if not self.full:
            return mean, F.softplus(raw) + 1e-4
        lower = raw.new_zeros(len(x), self.n, self.n)
        lower[:, self.indices[0], self.indices[1]] = raw
        diagonal = lower.diagonal(dim1=-2, dim2=-1)
        lower = lower + torch.diag_embed(F.softplus(diagonal) + 1e-4 - diagonal)
        return mean, lower

    def loss(self, x, y):
        mean, scale = self.distribution(x)
        if self.full:
            return -torch.distributions.MultivariateNormal(mean, scale_tril=scale).log_prob(y).mean() / self.n
        return (.5 * ((y - mean) / scale).square() + scale.log() + .5 * math.log(2 * math.pi)).mean()

    def forward(self, x, noise):
        mean, scale = self.distribution(x)
        return mean + (torch.bmm(scale, noise.unsqueeze(-1)).squeeze(-1) if self.full else scale * noise)


def channel_spec(name):
    preset = PAPER2309_PRESETS[name]
    value = dict(name=name, n=preset.n, message_dim=64 if name == "SSPA" else 16,
        ebno_db=preset.ebn0_db, rate=preset.rate, noise_std=ebno_to_noise(preset.ebn0_db, preset.rate),
        input_law="iid N(0,1) real coordinates; no minibatch normalization",
        representation="direct output", decoder_csi=False)
    if name == "SSPA":
        value.update(gain=5., a0=1.5, rapp_p=3., actual_real_noise_std=value["noise_std"] / math.sqrt(2))
    if name == "TDL":
        value["tdl_profile"] = tdl_effective_profile(symbols=preset.n // 2)
    return value


def physical_channel(name, x, ebno=None):
    spec = channel_spec(name)
    noise = spec["noise_std"] if ebno is None else ebno_to_noise(ebno, spec["rate"])
    return channel_registry()[name](x, noise, x.device)


class FastTrainer:
    def __init__(self, cfg, device, official, wflow=None):
        cfg.validate()
        self.cfg, self.device, self.official, self.wflow = cfg, torch.device(device), official, wflow
        self.spec = channel_spec(cfg.channel)
        self.n = self.spec["n"]
        set_seed(cfg.seed)
        self.latent_dim = max(16, self.n) if cfg.family in ("sinkhorn", "wgan") else self.n
        if cfg.family == "gaussian":
            self.model = GaussianNetwork(self.n, cfg.parameters, cfg.gaussian_covariance == "full").to(device)
        else:
            self.model = ChannelNetwork(self.n, cfg.parameters,
                cfg.family in ("diffusion", "flow", "consistency"), official.embedding, self.latent_dim,
                nn.ReLU if cfg.family == "wgan" else nn.SiLU).to(device)
        self.ema = deepcopy(self.model).requires_grad_(False)
        self.target = deepcopy(self.model).requires_grad_(False) if cfg.family == "consistency" else None
        if cfg.family == "sinkhorn":
            if wflow is None:
                raise ValueError("Pinned W-Flow source is required")
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                                               betas=(.9, .95), weight_decay=.01)
        elif cfg.family == "wgan":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=cfg.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.critic = mlp(2 * self.n, 1, cfg.parameters, nn.ReLU).to(device) if cfg.family == "wgan" else None
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=cfg.learning_rate) if self.critic is not None else None
        self.condition_rng = torch.Generator(device=device).manual_seed(cfg.seed + 1000000)
        self.update, self.train_seconds = 0, 0.
        self.counts = dict(oracle_outputs=0, input_anchors=0, generator_updates=0, critic_updates=0,
                           training_generated_outputs=0, target_network_outputs=0)
        self.ddim_cache = {}
        gd = official.gd
        self.diffusion_args = dict(betas=gd.get_named_beta_schedule("cosine", cfg.diffusion_steps),
            model_mean_type=gd.ModelMeanType.EPSILON, model_var_type=gd.ModelVarType.FIXED_SMALL,
            loss_type=gd.LossType.MSE, rescale_timesteps=False)
        self.diffusion = gd.GaussianDiffusion(**self.diffusion_args)
        self.consistency = official.cm.KarrasDenoiser(sigma_min=cfg.consistency_sigma_min,
            sigma_max=cfg.consistency_sigma_max, sigma_data=cfg.consistency_sigma_data,
            rho=7., weight_schedule="uniform", distillation=True, loss_norm="l2")
        self.cm_schedule = official.cm.create_ema_and_scales_fn("adaptive", .95, "progressive", 2,
            cfg.consistency_end_scales, cfg.updates, 1)

    def step(self):
        c = self.cfg
        anchors = c.batch_size // c.samples_per_input if c.family == "sinkhorn" else c.batch_size
        x = torch.randn(anchors, self.n, device=self.device, generator=self.condition_rng)
        if c.family == "sinkhorn":
            x = x.repeat_interleave(c.samples_per_input, 0)
        y = physical_channel(c.channel, x)
        progress_step = min(self.update, c.updates)
        if c.train_seconds:
            progress_step = min(int(self.train_seconds / c.train_seconds * c.updates), c.updates)
        info, generator_update = {}, True
        if c.family == "gaussian":
            loss = self.model.loss(x, y)
        elif c.family == "flow":
            t = torch.rand(len(x), device=self.device)
            path = self.official.path.sample(x_0=torch.randn_like(y), x_1=y, t=t)
            loss = F.mse_loss(self.model(path.x_t, t * 1000, x), path.dx_t)
        elif c.family == "diffusion":
            t = torch.randint(c.diffusion_steps, (len(x),), device=self.device)
            loss = self.diffusion.training_losses(self.model, y, t, model_kwargs={"condition": x})["loss"].mean()
        elif c.family == "consistency":
            target_decay, scales = self.cm_schedule(progress_step)
            loss = self.consistency.consistency_losses(self.model, y, scales,
                model_kwargs={"condition": x}, target_model=self.target)["loss"].mean()
            info.update(target_ema=target_decay, scales=scales)
            self.counts["target_network_outputs"] += len(x)
        elif c.family == "sinkhorn":
            self.optimizer.param_groups[0]["lr"] = 1e-6 + (c.learning_rate - 1e-6) * min(self.update / 5000, 1.)
            g = self.model(torch.randn(len(x), self.latent_dim, device=self.device), condition=x)
            with torch.no_grad():
                r = self.model(torch.randn(len(x), self.latent_dim, device=self.device), condition=x)
            g, p, r = [a.reshape(anchors, c.samples_per_input, self.n) for a in (g, y, r)]
            losses, _ = self.wflow.drift_loss_ot(g, p, r, R_list=(c.sinkhorn_r,),
                sinkhorn_num_iter=c.sinkhorn_iterations, disable_diag_mask=True, use_quadratic_cost=True)
            loss = losses.mean()
            self.counts["training_generated_outputs"] += 2 * len(x)
        else:
            with torch.no_grad():
                fake = self.model(torch.randn(len(x), self.latent_dim, device=self.device), condition=x)
            loss_d = self.critic(torch.cat((fake, x), -1)).mean() - self.critic(torch.cat((y, x), -1)).mean()
            self.critic_optimizer.zero_grad()
            loss_d.backward()
            if not torch.isfinite(loss_d) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.critic.parameters()):
                raise FloatingPointError("Nonfinite critic update")
            self.critic_optimizer.step()
            with torch.no_grad():
                for p in self.critic.parameters():
                    p.clamp_(-c.critic_clip, c.critic_clip)
            self.counts["critic_updates"] += 1
            self.counts["training_generated_outputs"] += len(x)
            info["critic_loss"] = loss_d.detach().item()
            generator_update = (self.update + 1) % c.critic_steps == 0
            loss = loss_d.detach()
            if generator_update:
                self.critic.requires_grad_(False)
                fake = self.model(torch.randn(len(x), self.latent_dim, device=self.device), condition=x)
                loss = -self.critic(torch.cat((fake, x), -1)).mean()
                self.counts["training_generated_outputs"] += len(x)
        if generator_update:
            self.optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 2. if c.family == "sinkhorn" else 1., error_if_nonfinite=True)
            self.optimizer.step()
            with torch.no_grad():
                for target, value in zip(self.ema.parameters(), self.model.parameters()):
                    target.lerp_(value, 1 - c.ema_decay)
                if self.target is not None:
                    for target, value in zip(self.target.parameters(), self.model.parameters()):
                        target.lerp_(value, 1 - target_decay)
            self.counts["generator_updates"] += 1
            info["gradient_norm"] = norm.item()
            if self.critic is not None:
                self.critic.requires_grad_(True)
        self.counts["oracle_outputs"] += len(x)
        self.counts["input_anchors"] += anchors
        self.update += 1
        info.update(update=self.update, loss=loss.detach().item())
        if not math.isfinite(info["loss"]):
            raise FloatingPointError("Nonfinite training loss")
        return info

    def sample(self, x, steps=1, noise=None, model=None):
        c = self.cfg
        model = (self.ema if c.inference_weights == "ema" else self.model) if model is None else model
        if noise is None:
            noise = torch.randn(len(x), self.latent_dim, device=x.device, dtype=x.dtype)
        if c.family == "gaussian":
            if steps != 1:
                raise ValueError("Gaussian generator is one-shot")
            return model(x, noise)
        if c.family in ("sinkhorn", "wgan"):
            if steps != 1:
                raise ValueError("Direct generator is one-shot")
            return model(noise, condition=x)
        if steps < 1:
            raise ValueError("Positive NFE required")
        if c.family == "flow":
            state = noise
            for i in range(steps):
                t = x.new_full((len(x),), 1000 * i / steps)
                state = state + model(state, t, x) / steps
            return state
        if c.family == "consistency":
            cm = self.official.cm
            denoise = lambda state, sigma: self.consistency.denoise(model, state, sigma, condition=x)[1]
            state = noise * c.consistency_sigma_max
            sigmas = cm.get_sigmas_karras(40, c.consistency_sigma_min, c.consistency_sigma_max, device=x.device)
            if steps == 1:
                return cm.sample_onestep.__wrapped__(denoise, state, sigmas)
            if steps != 2:
                raise ValueError("Consistency adapter supports one or two evaluations")
            return cm.stochastic_iterative_sampler.__wrapped__(denoise, state, sigmas, torch,
                ts=(0, c.consistency_two_step_index, 39), t_min=c.consistency_sigma_min,
                t_max=c.consistency_sigma_max, rho=7., steps=40)
        if steps not in self.ddim_cache:
            selected = self.official.respace.space_timesteps(c.diffusion_steps, f"ddim{steps}")
            self.ddim_cache[steps] = self.official.respace.SpacedDiffusion(use_timesteps=selected, **self.diffusion_args)
        diffusion = self.ddim_cache[steps]
        state = noise
        # Use the upstream single-step routine without its outer inference-only loop.
        for index in reversed(range(diffusion.num_timesteps)):
            t = torch.full((len(x),), index, device=x.device, dtype=torch.long)
            state = diffusion.ddim_sample(model, state, t, clip_denoised=False,
                model_kwargs={"condition": x}, eta=0.)["sample"]
        return state

    def state(self):
        return dict(config=asdict(self.cfg), model=self.model.state_dict(), ema=self.ema.state_dict(),
            target=None if self.target is None else self.target.state_dict(), optimizer=self.optimizer.state_dict(),
            critic=None if self.critic is None else self.critic.state_dict(),
            critic_optimizer=None if self.critic_optimizer is None else self.critic_optimizer.state_dict(),
            update=self.update, counts=self.counts, train_seconds=self.train_seconds,
            condition_rng=self.condition_rng.get_state(), rng=rng_state())

    def restore(self, state):
        if state["config"] != asdict(self.cfg):
            raise ValueError("Baseline checkpoint configuration mismatch")
        for name in ("model", "ema", "target", "optimizer", "critic", "critic_optimizer"):
            if getattr(self, name) is not None:
                getattr(self, name).load_state_dict(state[name])
        self.update, self.counts, self.train_seconds = state["update"], state["counts"], state["train_seconds"]
        self.condition_rng.set_state(state["condition_rng"].cpu())
        restore_rng(state["rng"])


class FastImplant(nn.Module):
    def __init__(self, trainer, steps=1):
        super().__init__()
        self.trainer, self.steps = trainer, steps
        self.model = deepcopy(trainer.ema if trainer.cfg.inference_weights == "ema" else trainer.model)
        self.model.eval().requires_grad_(False)

    def forward(self, x):
        return self.trainer.sample(x, self.steps, model=self.model)
