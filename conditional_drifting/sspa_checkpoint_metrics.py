"""Read-only conditional moment derivatives for row-independent generators."""

import torch


def output_jacobians(outputs, inputs):
    return torch.stack([
        torch.autograd.grad(outputs[:, j].sum(), inputs,
                            retain_graph=j + 1 < outputs.shape[1])[0]
        for j in range(outputs.shape[1])
    ], dim=1)


def moment_derivatives(outputs, jacobians):
    """Unbiased sample covariance and its pathwise input derivative."""
    if outputs.ndim != 3 or jacobians.ndim != 4:
        raise ValueError("Expected anchor/sample/output[/input] axes")
    if outputs.shape != jacobians.shape[:3] or outputs.shape[1] < 2:
        raise ValueError("Matching clouds with at least two samples required")
    mean = outputs.mean(1)
    mean_jacobian = jacobians.mean(1)
    centered = outputs - mean[:, None]
    centered_jacobian = jacobians - mean_jacobian[:, None]
    covariance = centered.transpose(1, 2) @ centered / (outputs.shape[1] - 1)
    term = torch.einsum("anik,anj->aijk", centered_jacobian, centered)
    covariance_jacobian = (term + term.transpose(1, 2)) / (outputs.shape[1] - 1)
    return mean, covariance, mean_jacobian, covariance_jacobian


def summarize_moments(outputs, jacobians, true_mean, true_jacobian, noise_std):
    mean, covariance, mean_jacobian, covariance_jacobian = moment_derivatives(outputs, jacobians)
    true_covariance = torch.eye(outputs.shape[-1], device=outputs.device) * noise_std ** 2 / 2
    jac_error = (mean_jacobian - true_jacobian).square().sum((1, 2)).mean().sqrt()
    jac_norm = true_jacobian.square().sum((1, 2)).mean().sqrt()
    return {
        "mean_error_l2": (mean - true_mean).norm(dim=1).mean().item(),
        "covariance_error_fro": (covariance - true_covariance).norm(dim=(1, 2)).mean().item(),
        "mean_jacobian_error_rms_fro": jac_error.item(),
        "mean_jacobian_relative_error": (jac_error / jac_norm.clamp_min(1e-12)).item(),
        "true_mean_jacobian_rms_fro": jac_norm.item(),
        "covariance_jacobian_rms_fro": covariance_jacobian.square().sum((1, 2, 3)).mean().sqrt().item(),
        "sample_jacobian_rms_fro": jacobians.square().sum((2, 3)).mean().sqrt().item(),
    }
