import torch

from conditional_drifting.baselines.diffusion import (
    ConditionalDiffusionMLP,
    DiffusionConfig,
    build_ddim_trajectory,
    evaluate_diffusion_model,
    cosine_beta_schedule,
    resolve_learning_rate_schedule,
)
from conditional_drifting.baselines.gan import ConditionalGANGenerator
from conditional_drifting.baselines.paper_wgan import PaperWGANDiscriminator, PaperWGANGenerator


def test_cosine_schedule_has_expected_shape():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert torch.all(betas > 0)


def test_optional_baseline_models_preserve_shapes():
    x = torch.randn(32, 2)
    t = torch.randint(0, 100, (32,))
    diffusion = ConditionalDiffusionMLP(n=2, num_steps=100)
    gan = ConditionalGANGenerator(n=2)
    paper_wgan_g = PaperWGANGenerator(n=2, hidden_dim=16)
    paper_wgan_d = PaperWGANDiscriminator(n=2, hidden_dim=16)
    assert diffusion(x, t, x).shape == x.shape
    assert gan(x).shape == x.shape
    assert gan.sample_y(x).shape == x.shape
    assert paper_wgan_g(x).shape == x.shape
    assert paper_wgan_d(x, x).shape == (x.shape[0], 1)


def test_chunked_diffusion_eval_reports_all_chunks():
    model = ConditionalDiffusionMLP(n=2, hidden_dim=8, num_steps=4)
    model._diffusion_state = {
        "betas": cosine_beta_schedule(4),
        "alphas": 1.0 - cosine_beta_schedule(4),
        "alphas_prod": torch.cumprod(1.0 - cosine_beta_schedule(4), dim=0),
        "alphas_bar_sqrt": torch.sqrt(torch.cumprod(1.0 - cosine_beta_schedule(4), dim=0)),
        "one_minus_alphas_bar_sqrt": torch.sqrt(1.0 - torch.cumprod(1.0 - cosine_beta_schedule(4), dim=0)),
        "pred_type": "epsilon",
        "is_residual": True,
    }

    cfg = DiffusionConfig(n=2, eval_size=10, eval_batch_size=4, num_steps=4, swd_projections=8)
    seen = []

    def identity_channel(x, noise_std, device):
        return x

    result = evaluate_diffusion_model(
        model,
        identity_channel,
        cfg,
        torch.device("cpu"),
        progress_callback=lambda done, total, chunk: seen.append((done, total, chunk["batch_size"])),
    )

    assert result["target_true"].shape == (10, 2)
    assert result["target_pred"].shape == (10, 2)
    assert len(result["chunk_timings"]) == 3
    assert seen == [(1, 3, 4), (2, 3, 4), (3, 3, 2)]


def test_ddim_trajectory_matches_skip_logic_when_divisible():
    assert build_ddim_trajectory(100, 100) == list(range(100))
    assert build_ddim_trajectory(100, 50)[:5] == [1, 3, 5, 7, 9]
    assert build_ddim_trajectory(100, 20)[:5] == [4, 9, 14, 19, 24]
    assert build_ddim_trajectory(100, 10) == [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]


def test_learning_rate_schedule_defaults_to_single_stage():
    cfg = DiffusionConfig(epochs=12, learning_rate=5e-4)
    assert resolve_learning_rate_schedule(cfg) == [(12, 5e-4)]


def test_learning_rate_schedule_requires_full_epoch_coverage():
    cfg = DiffusionConfig(epochs=12, learning_rate_schedule=((10, 1e-3), (1, 1e-4)))
    try:
        resolve_learning_rate_schedule(cfg)
    except ValueError as exc:
        assert "covers 11 epochs" in str(exc)
    else:
        raise AssertionError("expected ValueError for incomplete learning_rate_schedule")
