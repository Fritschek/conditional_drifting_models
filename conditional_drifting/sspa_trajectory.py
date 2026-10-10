"""Restricted raw-output SSPA trajectories for the revision stability study."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from .channels import sspa
from .losses import _batched_sinkhorn_barycentric_projection
from .metrics import (conditional_anchor_swd, conditional_anchor_gaussian_w2,
                      conditional_anchor_mean_l2, conditional_anchor_cov_fro,
                      sliced_wasserstein_distance)
from .model import ConditionalDriftingGenerator
from .training import set_seed
from .transport_reference import log_sinkhorn, marginal_residuals


POLICIES = ("legacy_separate_adaptive", "fixed_common", "shared_adaptive")


@dataclass(frozen=True)
class TrajectoryConfig:
    seed: int = 9001
    policy: str = "legacy_separate_adaptive"
    batch_size: int = 4096
    n: int = 8
    samples: int = 4
    hidden_dim: int = 128
    latent_dim: int = 16
    noise_std: float = 0.3250531435997416
    learning_rate: float = .001
    drift_scale: float = 1.
    max_drift_norm: float = 2.
    iterations: int = 10
    min_epsilon: float = .001
    calibration_seed: int = 800001
    calibration_anchors: int = 4096
    validation_seed: int = 800101
    validation_anchors: int = 128
    validation_samples: int = 128
    validation_projections: int = 64

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError("Unknown epsilon policy")
        if self.n != 8 or min(self.batch_size, self.samples, self.iterations,
                              self.calibration_anchors, self.validation_anchors) < 1:
            raise ValueError("This runner requires n=8 and positive cloud sizes")
        if self.validation_samples < 2:
            raise ValueError("Covariance validation needs at least two samples")


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def isolated_rng(seed):
    state = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ("sspa_trajectory.py", "channels.py", "losses.py", "metrics.py",
             "model.py", "training.py", "transport_reference.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def config_hash(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def pooled_epsilon(generated, positive, reference, minimum):
    costs = torch.cat([(.5 * torch.cdist(generated, other).square()).flatten()
                       for other in (positive, reference)])
    positive_costs = costs[costs > 0]
    return max(minimum, positive_costs.median().item()) if positive_costs.numel() else minimum


def field(generated, positive, reference, cfg, fixed_epsilon, diagnostics=False):
    shape = (-1, cfg.samples, cfg.n)
    g, p, r = (v.detach().float().reshape(shape) for v in (generated, positive, reference))
    epsilon = None
    if cfg.policy == "fixed_common":
        epsilon = fixed_epsilon
    elif cfg.policy == "shared_adaptive":
        epsilon = pooled_epsilon(g, p, r, cfg.min_epsilon)
    kwargs = dict(epsilon=epsilon, min_epsilon=cfg.min_epsilon,
                  iterations=cfg.iterations, return_diagnostics=diagnostics)
    cross = _batched_sinkhorn_barycentric_projection(g, p, p, **kwargs)
    own = _batched_sinkhorn_barycentric_projection(g, r, r, **kwargs)
    if diagnostics:
        cross, cross_info = cross
        own, own_info = own
    # Preserve the historical subtraction order for exact legacy comparisons.
    raw = ((cross - g) - (own - g)).reshape(-1, cfg.n)
    norms = raw.norm(dim=1, keepdim=True) + 1e-8
    drift = raw * torch.clamp(cfg.max_drift_norm / norms, max=1.)
    if not diagnostics:
        return drift, {}
    info = {"raw_drift_mean_norm": raw.norm(dim=1).mean().item(),
            "clipped_drift_mean_norm": drift.norm(dim=1).mean().item(),
            "drift_clip_fraction": (norms > cfg.max_drift_norm).float().mean().item()}
    for name, result in (("cross", cross_info), ("self", own_info)):
        info[name + "_epsilon"] = float(result["epsilon"])
        info[name + "_floor_fraction"] = result["kernel_floor_mask"].float().mean().item()
        info.update({name + "_" + k: v for k, v in marginal_residuals(result["coupling"]).items()})
        info[name + "_normalized_column_relative"] = marginal_residuals(
            result["row_weights"] / cfg.samples)["column_relative"]
    info["epsilon_ratio"] = info["cross_epsilon"] / info["self_epsilon"]
    return drift, info


class SSPATrajectory:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        set_seed(cfg.seed)
        self.model = ConditionalDriftingGenerator(cfg.n, cfg.n, cfg.latent_dim,
                                                  cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.update = 0
        self.counts = dict(training_anchors=0, training_oracle_outputs=0,
                           training_generated_outputs=0, training_reference_outputs=0,
                           calibration_oracle_outputs=0, validation_oracle_outputs=0,
                           diagnostic_oracle_outputs=0)
        self.train_seconds = 0.
        self.validation_seconds = 0.
        self.history, self.trace = [], []
        # Training-law oracle-only calibration, independent of all training and
        # validation draws. The same frozen scale is recorded for every policy.
        with isolated_rng(cfg.calibration_seed), torch.no_grad():
            x = torch.randn(cfg.calibration_anchors, cfg.n, device=self.device)
            x = x.repeat_interleave(cfg.samples, dim=0)
            a, b = [sspa(x, cfg.noise_std, self.device).reshape(-1, cfg.samples, cfg.n)
                    for _ in range(2)]
            costs = .5 * torch.cdist(a, b).square()
            self.fixed_epsilon = max(cfg.min_epsilon, costs[costs > 0].median().item())
        self.counts["calibration_oracle_outputs"] = 2 * len(x)

    def step(self, diagnostics=False):
        c = self.cfg
        x = torch.randn(c.batch_size, c.n, device=self.device).repeat_interleave(c.samples, dim=0)
        positive = sspa(x, c.noise_std, self.device)
        generated = self.model(x)
        with torch.no_grad():
            reference = self.model(x)
        drift, info = field(generated, positive, reference, c, self.fixed_epsilon, diagnostics)
        loss = F.mse_loss(generated, (generated + c.drift_scale * drift).detach())
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.update += 1
        self.counts["training_anchors"] += c.batch_size
        for name in ("oracle", "generated", "reference"):
            self.counts["training_" + name + "_outputs"] += len(x)
        if diagnostics:
            info.update(update=self.update, loss=loss.item(), gradient_norm=gradient_norm.item(),
                        gradient_clipped=bool(gradient_norm > 1.))
            self.trace.append(info)
        return info

    @torch.no_grad()
    def validate(self, reference_check=True):
        c, device = self.cfg, self.device
        was_training = self.model.training
        self.model.eval()
        try:
            with isolated_rng(c.validation_seed):
                x = torch.randn(c.validation_anchors, c.n, device=device)
            xx = x.repeat_interleave(c.validation_samples, dim=0)
            with isolated_rng(c.validation_seed + 1):
                true = sspa(xx, c.noise_std, device).reshape(len(x), c.validation_samples, c.n)
            with isolated_rng(c.validation_seed + 2):
                other = sspa(xx, c.noise_std, device).reshape_as(true)
            with isolated_rng(c.validation_seed + 3):
                generated = self.model(xx).reshape_as(true)
            with isolated_rng(c.validation_seed + 4):
                generated2 = self.model(xx).reshape_as(true)
            info = dict(update=self.update, train_seconds=self.train_seconds,
                        anchor_norm_mean=x.norm(dim=1).mean().item())
            for name, metric in (("anchor_swd", lambda a, b: conditional_anchor_swd(
                    a, b, num_projections=c.validation_projections, seed=c.validation_seed + 5)),
                                 ("anchor_gw2", conditional_anchor_gaussian_w2),
                                 ("anchor_mean_l2", conditional_anchor_mean_l2),
                                 ("anchor_cov_fro", conditional_anchor_cov_fro)):
                info[name] = metric(true, generated)
                info[name + "_floor"] = metric(true, other)
            info["global_swd"] = sliced_wasserstein_distance(true.flatten(0, 1), generated.flatten(0, 1),
                num_projections=c.validation_projections, seed=c.validation_seed + 6)
            centered = generated - generated.mean(dim=1, keepdim=True)
            covariance = centered.transpose(1, 2) @ centered / (c.validation_samples - 1)
            eigs = torch.linalg.eigvalsh(covariance)
            info.update(output_norm_mean=generated.norm(dim=2).mean().item(),
                        conditional_variance=covariance.diagonal(dim1=1, dim2=2).mean().item(),
                        covariance_eigenvalue_min=eigs.min().item(), covariance_eigenvalue_max=eigs.max().item(),
                        latent_pair_rms=(generated - generated2).square().mean().sqrt().item())
            self.counts["validation_oracle_outputs"] += 2 * len(xx)
            if reference_check:
                with isolated_rng(c.validation_seed + 7):
                    dx = x.repeat_interleave(c.samples, dim=0)
                    positive = sspa(dx, c.noise_std, device)
                    g, r = self.model(dx), self.model(dx)
                    _, diagnostic = field(g, positive, r, c, self.fixed_epsilon, True)
                info["transport"] = diagnostic
                self.counts["diagnostic_oracle_outputs"] += len(dx)
                reference_rows = []
                for name, target in (("cross", positive), ("self", r)):
                    epsilon = diagnostic[name + "_epsilon"]
                    g3, t3 = [v.reshape(-1, c.samples, c.n) for v in (g, target)]
                    practical = _batched_sinkhorn_barycentric_projection(
                        g3[:8], t3[:8], t3[:8], epsilon=epsilon,
                        min_epsilon=c.min_epsilon, iterations=c.iterations).cpu()
                    g3, t3 = g3.cpu(), t3.cpu()
                    for anchor in range(min(8, len(x))):
                        ref = log_sinkhorn(g3[anchor], t3[anchor], epsilon, require_convergence=False)
                        reference_rows.append(dict(term=name, anchor=anchor, converged=ref["converged"],
                            iterations=ref["iterations"], row_relative=ref["row_relative"],
                            column_relative=ref["column_relative"], barycenter_rms=(
                                (practical[anchor] - ref["barycenter"]).square().mean().sqrt().item()
                                if ref["converged"] else None)))
                info["reference_checks"] = reference_rows
            info["counts"] = dict(self.counts)
            self.history.append(info)
            return info
        finally:
            self.model.train(was_training)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(config=asdict(self.cfg), config_hash=config_hash(self.cfg), source_hashes=source_hashes(),
                     device=str(self.device), torch_version=str(torch.__version__), update=self.update,
                     model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), rng=rng_state(),
                     counts=self.counts, fixed_epsilon=self.fixed_epsilon, train_seconds=self.train_seconds,
                     validation_seconds=self.validation_seconds, history=self.history, trace=self.trace)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["config_hash"] != config_hash(self.cfg) or state["config"] != asdict(self.cfg):
            raise ValueError("Checkpoint configuration mismatch")
        if state["source_hashes"] != source_hashes():
            raise ValueError("Checkpoint source mismatch")
        if state["device"] != str(self.device) or state["torch_version"] != str(torch.__version__):
            raise ValueError("Exact resume requires the same device and Torch version")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for name in ("update", "counts", "fixed_epsilon", "train_seconds", "validation_seconds", "history", "trace"):
            setattr(self, name, state[name])
        restore_rng(state["rng"])
