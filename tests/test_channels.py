import torch

from conditional_drifting.channels import awgn, optfib, rayleigh_awgn, sspa


def test_channels_preserve_shape():
    device = torch.device("cpu")
    x = torch.randn(16, 2)
    assert awgn(x, 0.3, device).shape == x.shape
    assert rayleigh_awgn(x, 0.3, device).shape == x.shape
    assert sspa(x, 0.3, device).shape == x.shape
    assert optfib(x, 0.3, device).shape == x.shape
