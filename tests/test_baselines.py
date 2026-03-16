import torch

from conditional_drifting.baselines.diffusion import ConditionalDiffusionMLP, cosine_beta_schedule
from conditional_drifting.baselines.gan import ConditionalGANGenerator


def test_cosine_schedule_has_expected_shape():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert torch.all(betas > 0)


def test_optional_baseline_models_preserve_shapes():
    x = torch.randn(32, 2)
    t = torch.randint(0, 100, (32,))
    diffusion = ConditionalDiffusionMLP(n=2)
    gan = ConditionalGANGenerator(n=2)
    assert diffusion(x, t, x).shape == x.shape
    assert gan(x).shape == x.shape
