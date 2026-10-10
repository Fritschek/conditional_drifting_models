"""Small float64 transport reference for audits, not the training solver."""

import math

import torch


def quadratic_cost(source, target):
    return .5*(source[:, None]-target[None]).square().sum(-1)


def marginal_residuals(coupling):
    n, m = coupling.shape[-2:]
    return {
        "row_relative": (coupling.sum(-1)*n-1).abs().amax().item(),
        "column_relative": (coupling.sum(-2)*m-1).abs().amax().item(),
    }


def log_sinkhorn(source, target, epsilon, tolerance=1e-8, max_iterations=50000,
                 check_every=10, require_convergence=True):
    """Uniform-mass entropic OT using log scalings and both marginal tests.

    No kernel floor, denominator stabilizer, or final row renormalization.
    The input tensors are detached: objective gradients use the envelope rule.
    Failure is explicit, never interpreted as a high-accuracy solution.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Expected source/target matrices with common output dimension")
    if min(len(source), len(target)) < 1 or epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("Nonempty clouds and finite positive epsilon required")
    x, y = source.detach().double(), target.detach().to(source.device).double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Finite cloud coordinates required")
    cost = quadratic_cost(x, y)
    result = log_sinkhorn_cost(cost, epsilon, tolerance, max_iterations, check_every, require_convergence)
    return result | {"barycenter": len(x)*result["coupling"]@y}


def log_sinkhorn_cost(cost, epsilon, tolerance=1e-8, max_iterations=50000,
                      check_every=10, require_convergence=True):
    """Explicit-cost reference, also usable for auditing a modified kernel."""
    cost = cost.detach().double()
    if cost.ndim != 2 or min(cost.shape) < 1 or not torch.isfinite(cost).all():
        raise ValueError("A finite nonempty cost matrix is required")
    if not math.isfinite(epsilon) or min(epsilon, tolerance, max_iterations, check_every) <= 0:
        raise ValueError("Positive finite epsilon, tolerance and iteration limits required")
    n, m = cost.shape
    log_kernel = -cost/epsilon
    log_a, log_b = -math.log(n), -math.log(m)
    log_u, log_v = cost.new_zeros(n), cost.new_zeros(m)
    converged = False
    for iteration in range(1, max_iterations+1):
        log_u = log_a-torch.logsumexp(log_kernel+log_v[None], dim=1)
        log_v = log_b-torch.logsumexp(log_kernel+log_u[:, None], dim=0)
        if iteration % check_every == 0 or iteration == max_iterations:
            log_plan = log_u[:, None]+log_kernel+log_v[None]
            plan = log_plan.exp()
            residuals = marginal_residuals(plan)
            if max(residuals.values()) <= tolerance:
                converged = True
                break
    # KL is relative to the product of uniform masses, not unnormalized entropy.
    objective = (plan*cost).sum()+epsilon*(plan*(log_plan-log_a-log_b)).sum()
    result = {"coupling": plan, "log_coupling": log_plan, "cost": cost,
              "objective": objective.item(),
              "epsilon": float(epsilon), "iterations": iteration,
              "converged": converged, **residuals}
    if require_convergence and not converged:
        raise RuntimeError(f"Reference did not converge: {residuals}, epsilon={epsilon}")
    return result


def sinkhorn_divergence(source, target, epsilon, **kwargs):
    cross = log_sinkhorn(source, target, epsilon, **kwargs)
    self_source = log_sinkhorn(source, source, epsilon, **kwargs)
    self_target = log_sinkhorn(target, target, epsilon, **kwargs)
    return cross["objective"]-.5*self_source["objective"]-.5*self_target["objective"]


def envelope_divergence(source, target, epsilon, **kwargs):
    """Differentiable cost envelope at freshly solved, detached optimal plans.

    The omitted entropy term is constant for the position derivative at the
    optimum. This returns the correct first derivative, not the objective value.
    Both occurrences of source in its self cost must remain differentiable.
    """
    cross = log_sinkhorn(source, target, epsilon, **kwargs)["coupling"]
    self_source = log_sinkhorn(source, source, epsilon, **kwargs)["coupling"]
    self_target = log_sinkhorn(target, target, epsilon, **kwargs)["coupling"]
    return ((cross*quadratic_cost(source, target)).sum()
            -.5*(self_source*quadratic_cost(source, source)).sum()
            -.5*(self_target*quadratic_cost(target, target)).sum())
