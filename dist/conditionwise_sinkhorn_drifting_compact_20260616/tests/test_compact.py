import torch

from conditional_drifting.channels import get_channel
from conditional_drifting.losses import direct_kernel_loss, joint_sinkhorn_loss
from conditional_drifting.model import ConditionalGenerator


def test_channels_shape():
    for name in ["AWGN", "Rayleigh", "SSPA"]:
        x = torch.randn(16, 2)
        assert get_channel(name)(x, 0.3).shape == x.shape
    x = torch.randn(16, 8)
    assert get_channel("TDL")(x, 0.3).shape == x.shape


def test_model_shape():
    model = ConditionalGenerator(2, 2)
    x = torch.randn(8, 2)
    assert model(x).shape == x.shape


def test_losses_are_finite():
    x = torch.randn(16, 2)
    y_model = x + 0.1 * torch.randn_like(x)
    y_true = x + 0.2 * torch.randn_like(x)
    for loss in [direct_kernel_loss(y_model, y_true)[0], joint_sinkhorn_loss(x, y_model, y_true)[0]]:
        assert torch.isfinite(loss)

