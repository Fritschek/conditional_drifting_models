"""Explicit child interventions; historical exact-resume code is unchanged."""

import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from .channels import sspa
from .losses import _batched_sinkhorn_barycentric_projection
from .model import ConditionalDriftingGenerator
from .sspa_solver_replay import barycenter, clipped_field, log_plan, residuals, statistics
from .sspa_trajectory import (SSPATrajectory, TrajectoryConfig, config_hash, isolated_rng,
                              restore_rng, rng_state, source_hashes)
from .sspa_update_screen import coordinate_scale
from .training import set_seed
from .transport_reference import log_sinkhorn

BRANCHES = ("control", "log_floor10", "log10", "log100", "log10_beta0",
            "log10_lr01", "log10_drift01", "official64")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Intervention(SSPATrajectory):
    def __init__(self, parent, branch, device, official, contract):
        if branch not in BRANCHES:
            raise ValueError("Unknown branch")
        state = torch.load(parent, map_location="cpu", weights_only=False)
        cfg = TrajectoryConfig(**state["config"])
        if state["config_hash"] != config_hash(cfg) or state["source_hashes"] != source_hashes():
            raise ValueError("Parent config/source mismatch")
        if state["device"] != str(device) or state["torch_version"] != str(torch.__version__):
            raise ValueError("Parent device/Torch mismatch")
        if cfg.policy != "fixed_common":
            raise ValueError("Fixed-common parent required")
        self.device, self.branch, self.official, self.contract = torch.device(device), branch, official, contract
        self.parent = dict(path=str(parent), sha256=digest(parent), update=state["update"],
                           config=state["config"], counts=state["counts"])
        self.cfg = replace(cfg, learning_rate=.0001) if branch == "log10_lr01" else cfg
        if branch == "log10_drift01":
            self.cfg = replace(cfg, drift_scale=.1)
        if branch == "official64":
            self.cfg = replace(cfg, batch_size=cfg.batch_size * cfg.samples // 64,
                               samples=64, learning_rate=.0002)
        set_seed(cfg.seed)
        self.model = ConditionalDriftingGenerator(cfg.n, cfg.n, cfg.latent_dim, cfg.hidden_dim).to(device)
        self.model.load_state_dict(state["model"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        if branch == "log10_lr01":
            self.optimizer.param_groups[0]["lr"] = .0001
        elif branch == "log10_beta0":
            self.optimizer.param_groups[0]["betas"] = (0., self.optimizer.param_groups[0]["betas"][1])
        elif branch == "official64":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6, betas=(.9, .95), weight_decay=.01)
        self.ema = copy.deepcopy(self.model) if branch == "official64" else None
        self.update, self.fixed_epsilon = state["update"], state["fixed_epsilon"]
        self.counts = copy.deepcopy(state["counts"])
        self.train_seconds, self.validation_seconds = 0., 0.
        self.history, self.trace = [], []
        self.parent_validation = next(r for r in state["history"] if r["update"] == state["update"])
        restore_rng(state["rng"])

    @torch.no_grad()
    def transport(self, g, p, r):
        epsilon = self.fixed_epsilon
        scale = coordinate_scale(g, p, r) if self.branch == "official64" else g.new_tensor(1.)
        if self.branch == "official64":
            epsilon = .05 * self.cfg.n * scale.square().item()
        plans, centers, floors = [], [], []
        for target in (p, r):
            if self.branch == "control":
                center, info = _batched_sinkhorn_barycentric_projection(g, target, target,
                    epsilon=epsilon, min_epsilon=.001, iterations=10, return_diagnostics=True)
                plan, floor = info["coupling"], info["kernel_floor_mask"]
            elif self.branch == "log_floor10":
                cost = .5 * torch.cdist(g, target).square() / epsilon
                cost = cost - cost.amin(-1, keepdim=True)
                kernel = (-cost).exp()
                floor = kernel < 1e-8
                plan = log_plan(kernel.clamp_min(1e-8).log(), 10)["plan"]
                center = barycenter(plan, target)
            else:
                reg = .05 * self.cfg.n if self.branch == "official64" else epsilon
                plan = self.official._sinkhorn_plan_log_impl(g / scale, target / scale, g.new_tensor(reg),
                    num_iter=100 if self.branch == "log100" else 10, diag_mask=False, use_quadratic_cost=True)
                center = torch.bmm(plan, target) / plan.sum(-1, keepdim=True).clamp_min(1e-12)
                floor = torch.zeros_like(plan, dtype=torch.bool)
            plans.append(plan)
            centers.append(center)
            floors.append(floor)
        raw, clipped = clipped_field(g, *centers)
        return raw, clipped, plans, floors, epsilon, scale

    def step(self, diagnostics=False):
        if self.branch == "control":
            return super().step(diagnostics)
        c = self.cfg
        x = torch.randn(c.batch_size, c.n, device=self.device).repeat_interleave(c.samples, 0)
        positive = sspa(x, c.noise_std, self.device)
        generated = self.model(x)
        with torch.no_grad():
            reference = self.model(x)
        g, p, r = [v.reshape(c.batch_size, c.samples, c.n) for v in (generated, positive, reference)]
        info = {}
        if self.branch == "official64":
            elapsed = self.update - self.parent["update"]
            self.optimizer.param_groups[0]["lr"] = 1e-6 + (.0002 - 1e-6) * min(elapsed / 5000, 1.)
            values, details = self.official.drift_loss_ot(g, p, r, R_list=(.05,), sinkhorn_num_iter=10,
                disable_diag_mask=True, use_quadratic_cost=True)
            loss = values.mean()
            if diagnostics:
                info.update(feature_scale=details["scale"].item(), force_squared=details["loss_0.05"].item())
        else:
            raw, drift, plans, floors, epsilon, _ = self.transport(g.detach(), p, r)
            loss = F.mse_loss(generated, (generated + c.drift_scale * drift.flatten(0, 1)).detach())
            if diagnostics:
                info.update(epsilon=epsilon, floor_fraction=torch.stack(floors).float().mean().item(),
                    max_marginal_relative_error=max(residuals(plan).max().item() for plan in plans),
                    drift_clip_fraction=(raw.norm(dim=-1) > 2.).float().mean().item())
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2. if self.branch == "official64" else 1., error_if_nonfinite=True)
        self.optimizer.step()
        if self.ema is not None:
            with torch.no_grad():
                for target, value in zip(self.ema.parameters(), self.model.parameters()):
                    target.mul_(.999).add_(value, alpha=.001)
        self.update += 1
        self.counts["training_anchors"] += c.batch_size
        for name in ("oracle", "generated", "reference"):
            self.counts["training_" + name + "_outputs"] += len(x)
        if diagnostics:
            info.update(update=self.update, loss=loss.item(), gradient_norm=norm.item(),
                        learning_rate=self.optimizer.param_groups[0]["lr"])
            self.trace.append(info)
        return info

    @torch.no_grad()
    def diagnose(self, reference_check=False):
        c = self.cfg
        with isolated_rng(800108):
            x = torch.randn(128, c.n, device=self.device).repeat_interleave(c.samples, 0)
            p, g, r = sspa(x, c.noise_std, self.device), self.model(x), self.model(x)
        g, p, r = [v.reshape(128, c.samples, c.n) for v in (g, p, r)]
        raw, clipped, plans, floors, epsilon, scale = self.transport(g, p, r)
        info = dict(epsilon=epsilon, feature_scale=scale.item(), raw_field_rms=raw.square().mean().sqrt().item(),
                    floor_fraction=torch.stack(floors).float().mean().item())
        for term, plan in zip(("cross", "self"), plans):
            errors = residuals(plan).amax(-1)
            info[term] = statistics(errors) | dict(fraction_above_1e_3=(errors > .001).float().mean().item())
        self.counts["diagnostic_oracle_outputs"] += len(x)
        if reference_check:
            info["reference"] = []
            for term, target, plan in zip(("cross", "self"), (p, r), plans):
                practical = (barycenter(plan, target) if self.branch in ("control", "log_floor10") else
                             torch.bmm(plan, target) / plan.sum(-1, keepdim=True).clamp_min(1e-12))
                for anchor in range(4):
                    ref = log_sinkhorn(g[anchor].cpu(), target[anchor].cpu(), epsilon,
                                       max_iterations=10000, require_convergence=False)
                    info["reference"].append(dict(term=term, anchor=anchor, converged=ref["converged"],
                        iterations=ref["iterations"], row_relative=ref["row_relative"], column_relative=ref["column_relative"],
                        barycenter_rms=(practical[anchor].cpu().double() - ref["barycenter"]).square().mean().sqrt().item()
                                       if ref["converged"] else None))
        return info

    def validate_child(self, reference_check=False):
        row = super().validate(reference_check=False)
        row["transport"] = self.diagnose(reference_check)
        if self.ema is not None:
            original, history = self.model, self.history
            self.model, self.history = self.ema, []
            try:
                row["ema"] = super().validate(reference_check=False)
            finally:
                self.model, self.history = original, history
        row["counts"] = dict(self.counts)
        return row

    def independent_panel(self):
        original_cfg, history, model = self.cfg, self.history, self.model
        self.cfg = replace(self.cfg, validation_seed=800501, validation_anchors=256,
                           validation_samples=512, validation_projections=128)
        self.history = []
        try:
            result = dict(raw=super().validate(reference_check=False))
            if self.ema is not None:
                self.model = self.ema
                result["ema"] = super().validate(reference_check=False)
            return result
        finally:
            self.cfg, self.history, self.model = original_cfg, history, model

    def save_child(self, path):
        state = dict(contract=self.contract, parent=self.parent, branch=self.branch, config=asdict(self.cfg),
                     model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     ema=self.ema.state_dict() if self.ema is not None else None, rng=rng_state(),
                     update=self.update, counts=self.counts, fixed_epsilon=self.fixed_epsilon,
                     history=self.history, trace=self.trace, train_seconds=self.train_seconds,
                     validation_seconds=self.validation_seconds, parent_validation=self.parent_validation)
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_child(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["contract"] != self.contract or state["branch"] != self.branch or state["parent"] != self.parent:
            raise ValueError("Child intervention/parent contract mismatch")
        if state["config"] != asdict(self.cfg):
            raise ValueError("Child config mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        for key in ("update", "counts", "fixed_epsilon", "history", "trace", "train_seconds", "validation_seconds", "parent_validation"):
            setattr(self, key, state[key])
        restore_rng(state["rng"])


def passes_gate(history, parent, noise_std, ema=False):
    selected = [r for r in history if r["update"] in (180000, 190000, 200000)]
    if len(selected) != 3:
        return False
    for row in selected:
        row = row.get("ema") if ema else row
        if row is None or not (row["anchor_swd"] <= 2 * max(parent["anchor_swd"], parent["anchor_swd_floor"])
            and .5 * noise_std ** 2 / 2 <= row["conditional_variance"] <= 2 * noise_std ** 2 / 2
            and row["anchor_mean_l2"] <= 2 * max(parent["anchor_mean_l2"], parent["anchor_mean_l2_floor"])):
            return False
    return True
