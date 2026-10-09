import torch

from conditional_drifting.channels import sspa
from conditional_drifting.symbolic_ae import SymbolicDecoder
from scripts.run_local_gradient_fidelity import (
    alignment, encoder_gradient, expected_loss, finite_difference_check, normalized_codebook, physical_channel,
)


def test_balanced_gradient_chunking():
    torch.manual_seed(1)
    codes = torch.randn(4, 2, dtype=torch.float64)
    decoder = torch.nn.Linear(2, 4).double()
    first, stats = expected_loss(codes, decoder, lambda x: x, 13, 8, 2)
    second, other = expected_loss(codes, decoder, lambda x: x, 13, 256, 2)
    torch.testing.assert_close(first, second)
    assert abs(stats["ce"] - other["ce"]) < 1e-12
    assert stats["trials"] == 52


def test_power_constraint_and_gradient():
    encoder = torch.nn.Linear(4, 2)
    codes = normalized_codebook(encoder, torch.eye(4))
    torch.testing.assert_close(codes.mean(), torch.tensor(0.), atol=1e-7, rtol=0)
    torch.testing.assert_close(codes.square().mean(), torch.tensor(1.))
    codes[0, 0].backward()
    assert encoder.weight.grad.norm() > 0


def test_physical_sspa_matches_repository():
    x = torch.randn(8, 4)
    torch.manual_seed(33)
    actual = physical_channel("SSPA", .325)(x)
    torch.manual_seed(33)
    expected = sspa(x, .325, x.device)
    torch.testing.assert_close(actual, expected)


def test_analytic_finite_difference():
    codes = torch.tensor([[1., .3], [-1., -.3]])
    decoder = SymbolicDecoder(message_dim=2, code_dim=2, hidden_dim=4)
    for name in ("AWGN", "SSPA"):
        result = finite_difference_check(codes, decoder, physical_channel(name, .4))
        assert result["absolute_error"] < 1e-8


def test_alignment_and_zero_reference():
    x = torch.tensor([1., 2.])
    assert abs(alignment(x, x)["cosine"] - 1) < 1e-12
    assert alignment(x, x)["relative_error"] == 0
    assert alignment(-x, x)["cosine"] < -.999
    assert alignment(x, x * 0)["relative_error"] is None


def test_encoder_chain_rule():
    torch.manual_seed(12)
    encoder = torch.nn.Linear(4, 2)
    decoder = torch.nn.Linear(2, 4)
    messages = torch.eye(4)
    codes = normalized_codebook(encoder, messages)
    code_gradient, _ = expected_loss(codes, decoder, lambda x: x, 7, 12, 0)
    via_codebook = encoder_gradient(encoder, messages, code_gradient)
    loss = torch.nn.functional.cross_entropy(decoder(normalized_codebook(encoder, messages)), torch.arange(4))
    direct = torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(encoder.parameters()))])
    torch.testing.assert_close(via_codebook, direct)
