"""Sample-only conditional embedding values and finite input derivatives.

These are finite-panel estimates, not bounds on a population input supremum.
"""

import torch


def perturbed_inputs(x, step):
    if step <= 0:
        raise ValueError("The difference step must be positive.")
    eye = torch.eye(x.numel(), dtype=x.dtype, device=x.device) * step
    pairs = torch.stack((x + eye, x - eye), dim=1).flatten(0, 1)
    return torch.cat((x[None], pairs))


def difference_weights(dimension, step, device=None):
    """Columns represent base value and each centered derivative of P-Q."""
    queries = 1 + 2 * dimension
    weights = torch.zeros(2 * queries, dimension + 1, dtype=torch.float64, device=device)
    weights[0, 0], weights[queries, 0] = 1, -1
    for j in range(dimension):
        plus, minus = 1 + 2 * j, 2 + 2 * j
        weights[plus, j + 1] = weights[queries + minus, j + 1] = 1 / (2 * step)
        weights[minus, j + 1] = weights[queries + plus, j + 1] = -1 / (2 * step)
    return weights


def empirical_embedding_gram(clouds, bandwidths, groups_per_block=2):
    """Full RBF kernels, averaged across bandwidths, retaining V-statistic diagonals."""
    if clouds.ndim != 3 or clouds.shape[1] < 2:
        raise ValueError("Expected [cloud, samples>=2, output coordinate].")
    if not bandwidths or min(bandwidths) <= 0 or groups_per_block < 1:
        raise ValueError("Positive bandwidths and block size required.")
    clouds = clouds.double()
    groups, samples, dim = clouds.shape
    flat = clouds.reshape(-1, dim)
    norm = flat.square().sum(-1)
    gram = torch.empty(groups, groups, dtype=clouds.dtype, device=clouds.device)
    for start in range(0, groups, groups_per_block):
        count = min(groups_per_block, groups - start)
        rows = flat[start * samples:(start + count) * samples]
        distances = (rows.square().sum(-1, keepdim=True) + norm[None] - 2 * rows @ flat.T).clamp_min(0)
        block = torch.zeros_like(distances)
        for bandwidth in bandwidths:
            block += torch.exp(-distances / (2 * bandwidth**2)) / len(bandwidths)
        gram[start:start + count] = block.reshape(count, samples, groups, samples).mean(dim=(1, 3))
    return (gram + gram.T) / 2


def gaussian_embedding_gram(means, variances, bandwidths):
    """Exact RBF embedding inner products for isotropic Gaussian laws (zero variance allowed)."""
    means = means.double()
    variances = torch.as_tensor(variances, dtype=means.dtype, device=means.device).expand(len(means))
    if (variances < 0).any() or not bandwidths or min(bandwidths) <= 0:
        raise ValueError("Variances must be nonnegative and bandwidths positive.")
    distance = (means[:, None] - means[None]).square().sum(-1)
    gram = torch.zeros_like(distance)
    for bandwidth in bandwidths:
        total = bandwidth**2 + variances[:, None] + variances[None]
        gram += (bandwidth**2 / total).pow(means.shape[-1] / 2) * torch.exp(-distance / (2 * total)) / len(bandwidths)
    return gram


def embedding_scores(gram, dimension, step):
    weights = difference_weights(dimension, step, gram.device)
    if gram.shape != (len(weights), len(weights)):
        raise ValueError("Gram matrix does not match the perturbation layout.")
    matrix = weights.T @ gram.double() @ weights
    return embedding_matrix_scores(matrix)


def embedding_matrix_scores(matrix):
    """Read value and derivative norms from their feature inner products."""
    matrix = (matrix + matrix.T) / 2
    eigenvalues = torch.linalg.eigvalsh(matrix[1:, 1:])
    tolerance = 1e-7 * max(1., matrix.abs().max().item())
    if matrix[0, 0] < -tolerance or eigenvalues[0] < -tolerance:
        raise ValueError("Embedding Gram matrix lost positive semidefiniteness.")
    return {
        "mmd": matrix[0, 0].clamp_min(0).sqrt().item(),
        "embedding_derivative_op": eigenvalues[-1].clamp_min(0).sqrt().item(),
        "embedding_derivative_fro": matrix[1:, 1:].trace().clamp_min(0).sqrt().item(),
        "derivative_gram_min_eigenvalue": eigenvalues[0].item(),
    }


def polynomial_features(y):
    """Fixed features [y/sqrt(d), vec(yy^T)/d]; no decoder-fitted weights."""
    y = y.double()
    dimension = y.shape[-1]
    second = (y.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2) / dimension
    return torch.cat((y / dimension**.5, second), dim=-1)


def polynomial_embedding_gram(clouds):
    features = polynomial_features(clouds).mean(1)
    return features @ features.T


def sample_input_jacobians(implant, x, samples, seed):
    """Per-sample derivatives for a batch-independent simulator in eval mode.

    Summing output coordinates assumes no cross-row operations. A separate
    finite-difference check is required before using this on another implant.
    """
    torch.manual_seed(seed)
    inputs = x[None].repeat(samples, 1).detach().requires_grad_(True)
    with torch.enable_grad():
        outputs = implant(inputs)
        jacobian = torch.stack([
            torch.autograd.grad(outputs[:, j].sum(), inputs,
                                retain_graph=j < outputs.shape[1] - 1)[0]
            for j in range(outputs.shape[1])
        ], dim=1)
    return outputs.detach(), jacobian.detach()


def pathwise_embedding_matrices(p, q, jp, jq, bandwidths):
    """Exact derivatives of finite empirical mean embeddings at fixed noise.

    Population derivatives still have Monte Carlo error. This requires a
    differentiable simulator, and is not an estimator for passive measurements.
    Only value and derivative diagonal blocks are returned (no cross block).
    """
    if not bandwidths or min(bandwidths) <= 0:
        raise ValueError("Positive bandwidths required.")
    y, jac = torch.cat((p, q)).double(), torch.cat((jp, jq)).double()
    weights = torch.cat((torch.ones(len(p), device=y.device, dtype=y.dtype) / len(p),
                         -torch.ones(len(q), device=y.device, dtype=y.dtype) / len(q)))
    input_dim = jac.shape[-1]
    delta = y[:, None] - y[None]
    distance = delta.square().sum(-1)
    left = torch.einsum("ijo,ioa->ija", delta, jac)
    right = torch.einsum("ijo,joa->ija", delta, jac)
    rbf = torch.zeros(input_dim + 1, input_dim + 1, dtype=y.dtype, device=y.device)
    for bandwidth in bandwidths:
        kernel = torch.exp(-distance / (2 * bandwidth**2))
        signed = kernel * weights[:, None] * weights[None] / len(bandwidths)
        rbf[0, 0] += signed.sum()
        rbf[1:, 1:] += torch.einsum("ij,ioa,job->ab", signed, jac, jac) / bandwidth**2
        rbf[1:, 1:] -= torch.einsum("ij,ija,ijb->ab", signed, left, right) / bandwidth**4
    features = polynomial_features(y)
    output_dim = y.shape[-1]
    second_jac = (torch.einsum("noa,np->nopa", jac, y)
                  + torch.einsum("no,npa->nopa", y, jac)).flatten(1, 2) / output_dim
    feature_jac = torch.cat((jac / output_dim**.5, second_jac), dim=1)
    mean_difference = weights @ features
    derivative_difference = torch.einsum("n,nfa->fa", weights, feature_jac)
    moments = torch.zeros_like(rbf)
    moments[0, 0] = mean_difference.square().sum()
    moments[1:, 1:] = derivative_difference.T @ derivative_difference
    return {"rbf": rbf, "moments": moments, "augmented": rbf + moments}


def moment_scores(p, q, dimension, step):
    p, q = p.double(), q.double()
    means = p.mean(1) - q.mean(1)
    def covariance(cloud):
        centered = cloud - cloud.mean(1, keepdim=True)
        return centered.transpose(1, 2) @ centered / (cloud.shape[1] - 1)
    cov = covariance(p) - covariance(q)
    def derivatives(values):
        return ((values[1::2] - values[2::2]) / (2 * step)).reshape(dimension, -1).T
    return {
        "mean_l2": means[0].norm().item(),
        "covariance_fro": cov[0].norm().item(),
        "mean_derivative_op": torch.linalg.matrix_norm(derivatives(means), ord=2).item(),
        "covariance_derivative_op": torch.linalg.matrix_norm(derivatives(cov), ord=2).item(),
    }
