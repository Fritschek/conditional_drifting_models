"""Leave-one-component-out study of the successful W-Flow SSPA recipe."""

import copy
from dataclasses import replace

import torch
import torch.nn.functional as F

from .channels import sspa
from .sspa_intervention import Intervention
from .sspa_update_screen import coordinate_scale

VARIANTS = ("k4", "raw_loss64", "fixed_epsilon64", "old_optimizer64")


def component_loss(official, g, p, r, variant, fixed_epsilon):
    scale = coordinate_scale(g.detach(), p, r)
    reg = fixed_epsilon / (g.shape[-1] * scale.square().item()) if variant == "fixed_epsilon64" else .05
    if variant == "raw_loss64":
        with torch.no_grad():
            velocity = official._compute_V_debiased(
                g.detach() / scale, p / scale, r / scale,
                eps=g.new_tensor(.05 * g.shape[-1]), num_iter=10,
                neg_diag_mask=False, use_quadratic_cost=True)
            target = g.detach() + scale * velocity
        loss = F.mse_loss(g, target)
        force_squared = velocity.square().mean()
    else:
        values, details = official.drift_loss_ot(g, p, r, R_list=(reg,),
            sinkhorn_num_iter=10, disable_diag_mask=True, use_quadratic_cost=True)
        loss = values.mean()
        force_squared = details[f"loss_{reg}"]
    return loss, scale, force_squared, reg


class ComponentIntervention(Intervention):
    def __init__(self, parent, variant, device, official, contract):
        if variant not in (*VARIANTS, "official64"):
            raise ValueError("Unknown component variant")
        super().__init__(parent, "official64", device, official, contract)
        self.variant = variant
        if variant == "k4":
            self.cfg = replace(self.cfg, batch_size=self.cfg.batch_size * 16, samples=4)
        elif variant == "old_optimizer64":
            state = torch.load(parent, map_location="cpu", weights_only=False)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
            self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
            self.cfg = replace(self.cfg, learning_rate=.001)

    def step(self, diagnostics=False):
        c = self.cfg
        x = torch.randn(c.batch_size, c.n, device=self.device).repeat_interleave(c.samples, 0)
        positive = sspa(x, c.noise_std, self.device)
        generated = self.model(x)
        with torch.no_grad():
            reference = self.model(x)
        g, p, r = [v.reshape(c.batch_size, c.samples, c.n) for v in (generated, positive, reference)]
        if self.variant != "old_optimizer64":
            elapsed = self.update - self.parent["update"]
            self.optimizer.param_groups[0]["lr"] = 1e-6 + (.0002 - 1e-6) * min(elapsed / 5000, 1.)
        loss, scale, force_squared, reg = component_loss(self.official, g, p, r, self.variant, self.fixed_epsilon)
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2., error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for target, value in zip(self.ema.parameters(), self.model.parameters()):
                target.mul_(.999).add_(value, alpha=.001)
        self.update += 1
        self.counts["training_anchors"] += c.batch_size
        for name in ("oracle", "generated", "reference"):
            self.counts["training_" + name + "_outputs"] += len(x)
        info = {}
        if diagnostics:
            info = dict(update=self.update, loss=loss.item(), gradient_norm=norm.item(),
                coordinate_scale=scale.item(), force_squared=force_squared.item(),
                physical_epsilon=reg * c.n * scale.square().item(),
                learning_rate=self.optimizer.param_groups[0]["lr"])
            self.trace.append(info)
        return info

    def validate_child(self, reference_check=False):
        # Parent transport diagnostics assume R=.05. Avoid mislabelling the
        # fixed-epsilon branch; this study uses the identical law-only panel.
        from .sspa_trajectory import SSPATrajectory
        row = SSPATrajectory.validate(self, reference_check=False)
        original, history = self.model, self.history
        self.model, self.history = self.ema, []
        try:
            row["ema"] = SSPATrajectory.validate(self, reference_check=False)
        finally:
            self.model, self.history = original, history
        row["counts"] = dict(self.counts)
        return row

    def save_child(self, path):
        # The component name lives in the exact-resume contract, while the
        # inherited branch field records the recipe this intervention modifies.
        if self.contract.get("component") != self.variant:
            raise ValueError("Component must be recorded in resume contract")
        super().save_child(path)
