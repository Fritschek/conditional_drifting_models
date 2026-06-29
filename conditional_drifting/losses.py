import torch
import torch.nn.functional as F


def median_bandwidth(x, floor=0.2):
    with torch.no_grad():
        distances = torch.cdist(x.detach(), x.detach())
        values = distances[distances > 0]
        if values.numel() == 0:
            return floor
        return max(float(values.median().item()), floor)


def clamp_drift(drift, max_norm):
    if max_norm is None:
        return drift
    norm = drift.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return drift * scale


def gaussian_barycenter(source, target, bandwidth):
    cost = torch.cdist(source.detach(), target.detach()).square()
    weights = torch.exp(-cost / (2.0 * bandwidth * bandwidth))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights @ target.detach()


def sinkhorn_barycenter(source_features, target_features, target_values, epsilon, iterations):
    cost = 0.5 * torch.cdist(source_features.detach(), target_features.detach()).square()
    if epsilon is None:
        values = cost[cost > 0]
        epsilon = max(float(values.median().item()) if values.numel() else 1e-3, 1e-3)
    kernel = torch.exp(-(cost / epsilon - (cost / epsilon).amin(dim=1, keepdim=True))).clamp_min(1e-12)
    n_source = source_features.shape[0]
    n_target = target_features.shape[0]
    a = torch.full((n_source,), 1.0 / n_source, device=source_features.device)
    b = torch.full((n_target,), 1.0 / n_target, device=source_features.device)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iterations):
        u = a / (kernel @ v).clamp_min(1e-12)
        v = b / (kernel.t() @ u).clamp_min(1e-12)
    coupling = u[:, None] * kernel * v[None, :]
    rows = coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return rows @ target_values.detach()


def batched_sinkhorn_barycenter(source_features, target_features, target_values, epsilon, iterations):
    cost = 0.5 * torch.cdist(source_features.detach(), target_features.detach()).square()
    if epsilon is None:
        values = cost[cost > 0]
        epsilon = max(float(values.median().item()) if values.numel() else 1e-3, 1e-3)
    scaled = cost / epsilon
    kernel = torch.exp(-(scaled - scaled.amin(dim=2, keepdim=True))).clamp_min(1e-12)
    batch, n_source, _ = source_features.shape
    n_target = target_features.shape[1]
    a = torch.full((batch, n_source), 1.0 / n_source, device=source_features.device)
    b = torch.full((batch, n_target), 1.0 / n_target, device=source_features.device)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iterations):
        u = a / torch.bmm(kernel, v.unsqueeze(2)).squeeze(2).clamp_min(1e-12)
        v = b / torch.bmm(kernel.transpose(1, 2), u.unsqueeze(2)).squeeze(2).clamp_min(1e-12)
    coupling = u[:, :, None] * kernel * v[:, None, :]
    rows = coupling / coupling.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return torch.bmm(rows, target_values.detach())


def direct_kernel_loss(y_model, y_true, max_drift_norm=2.0, bandwidth_floor=0.2):
    bandwidth = median_bandwidth(torch.cat([y_model.detach(), y_true.detach()], dim=0), bandwidth_floor)
    target = gaussian_barycenter(y_model, y_true, bandwidth)
    drift = clamp_drift(target - y_model, max_drift_norm)
    return F.mse_loss(y_model, (y_model + drift).detach()), drift


def joint_sinkhorn_loss(x, y_model, y_true, iterations=10, max_drift_norm=2.0):
    source_features = torch.cat([x, y_model], dim=1)
    target_features = torch.cat([x, y_true], dim=1)
    target = sinkhorn_barycenter(source_features, target_features, y_true, None, iterations)
    drift = clamp_drift(target - y_model, max_drift_norm)
    return F.mse_loss(y_model, (y_model + drift).detach()), drift


def conditionwise_sinkhorn_loss(
    x_anchor,
    y_model_cloud,
    y_true_cloud,
    y_reference_cloud,
    generated_per_anchor,
    true_per_anchor,
    reference_per_anchor,
    iterations=10,
    max_drift_norm=2.0,
):
    batch = x_anchor.shape[0]
    y_model = y_model_cloud.reshape(batch, generated_per_anchor, -1)
    y_true = y_true_cloud.reshape(batch, true_per_anchor, -1)
    y_reference = y_reference_cloud.reshape(batch, reference_per_anchor, -1)
    positive_center = batched_sinkhorn_barycenter(y_model, y_true, y_true, None, iterations)
    self_center = batched_sinkhorn_barycenter(y_model, y_reference, y_reference, None, iterations)
    target = (y_model + (positive_center - y_model) - (self_center - y_model)).reshape_as(y_model_cloud)
    drift = clamp_drift(target - y_model_cloud, max_drift_norm)
    return F.mse_loss(y_model_cloud, (y_model_cloud + drift).detach()), drift
