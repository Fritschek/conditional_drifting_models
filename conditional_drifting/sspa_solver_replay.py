"""Frozen-cloud numerical comparisons; never used by the trajectory trainer."""

import math

import torch


def statistics(values):
    values = values.detach().double().flatten().cpu()
    if not len(values):
        return {"count": 0}
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite replay statistic")
    return dict(count=len(values), mean=values.mean().item(),
                sd=values.std(unbiased=False).item(), min=values.min().item(),
                median=values.median().item(), p90=torch.quantile(values, .9).item(),
                p99=torch.quantile(values, .99).item(), max=values.max().item())


def residuals(plan):
    return torch.stack(((plan.sum(-1) * plan.shape[-2] - 1).abs().amax(-1),
                        (plan.sum(-2) * plan.shape[-1] - 1).abs().amax(-1)), dim=-1)


def log_plan(log_kernel, iterations, denominator=0., tolerance=None, check_every=10):
    """Batched uniform OT. Freeze converged clouds; retain capped clouds explicitly.

    Initial log scalings are zero, matching production's unit scalings. Optional
    denominator reproduces the additive stabilizer in log arithmetic.
    """
    if log_kernel.ndim != 3 or not torch.isfinite(log_kernel).all():
        raise ValueError("Finite batched log kernels required")
    if iterations < 1 or check_every < 1 or denominator < 0:
        raise ValueError("Invalid solver limits")
    if tolerance is not None and tolerance <= 0:
        raise ValueError("Positive tolerance required")
    batch, n, m = log_kernel.shape
    log_a, log_b = -math.log(n), -math.log(m)
    active = torch.arange(batch, device=log_kernel.device)
    kernel = log_kernel
    u, v = kernel.new_zeros(batch, n), kernel.new_zeros(batch, m)
    output = torch.empty_like(log_kernel)
    counts = torch.full((batch,), iterations, dtype=torch.long, device=kernel.device)
    converged = torch.zeros(batch, dtype=torch.bool, device=kernel.device)
    stabilizer = kernel.new_tensor(math.log(denominator)) if denominator else None
    for step in range(1, iterations + 1):
        rows = torch.logsumexp(kernel + v[:, None, :], dim=2)
        u = log_a - (torch.logaddexp(rows, stabilizer) if stabilizer is not None else rows)
        cols = torch.logsumexp(kernel + u[:, :, None], dim=1)
        v = log_b - (torch.logaddexp(cols, stabilizer) if stabilizer is not None else cols)
        if step == iterations or (tolerance is not None and step % check_every == 0):
            plan = (u[:, :, None] + kernel + v[:, None, :]).exp()
            if tolerance is None:
                output[active] = plan
                break
            passed = residuals(plan).amax(-1) <= tolerance
            if passed.any():
                ids = active[passed]
                output[ids], counts[ids], converged[ids] = plan[passed], step, True
            if step == iterations:
                output[active[~passed]] = plan[~passed]
                break
            keep = ~passed
            if not keep.any():
                break
            active, kernel, u, v = active[keep], kernel[keep], u[keep], v[keep]
    return dict(plan=output, iterations=counts, converged=converged,
                residuals=residuals(output))


def barycenter(plan, target, readout="production"):
    target = target.to(plan)
    if readout == "production":
        weights = plan / (plan.sum(-1, keepdim=True) + 1e-8)
    elif readout == "row":
        weights = plan / plan.sum(-1, keepdim=True)
    elif readout == "uniform":
        weights = plan * plan.shape[-2]
    else:
        raise ValueError("Unknown readout")
    return weights @ target


def clipped_field(generated, cross, own, cap=2.):
    # Keep the trainer's float32 subtraction and radial-clipping conventions.
    raw = ((cross.float() - generated) - (own.float() - generated))
    norms = raw.norm(dim=-1, keepdim=True) + 1e-8
    return raw, raw * (cap / norms).clamp(max=1.)


def variance_terms(source, velocity, scale=1.):
    source, velocity = source.double(), velocity.double()
    centered = source - source.mean(1, keepdim=True)
    vc = velocity - velocity.mean(1, keepdim=True)
    first = 2 * scale * (centered * vc).mean((1, 2))
    second = scale ** 2 * vc.square().mean((1, 2))
    actual = (centered + scale * vc).square().mean((1, 2)) - centered.square().mean((1, 2))
    return dict(first=first, second=second, total=first + second, actual=actual)
