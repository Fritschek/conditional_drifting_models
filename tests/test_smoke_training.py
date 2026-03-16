import torch

from conditional_drifting.benchmark import BenchmarkConfig
from conditional_drifting.channels import channel_registry
from conditional_drifting.training import evaluate_residual_model, train_conditional_drifting


def test_drifting_smoke_train_and_eval():
    device = torch.device("cpu")
    cfg = BenchmarkConfig(dataset_size=256, batch_size=64, epochs=1, eval_size=128)
    channel_fn = channel_registry()["AWGN"]
    model, artifacts = train_conditional_drifting(channel_fn, cfg, device)
    result = evaluate_residual_model(model, channel_fn, cfg, device, metric_seed=11)
    assert len(artifacts.history) == 1
    assert result["swd"] >= 0.0
