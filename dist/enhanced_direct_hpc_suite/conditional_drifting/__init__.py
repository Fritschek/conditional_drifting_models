from __future__ import annotations

from importlib import import_module

__all__ = [
    "BenchmarkConfig",
    "ConditionalDriftingGenerator",
    "DriftingConfig",
    "baselines",
    "channel_registry",
    "evaluate_residual_model",
    "run_single_channel_benchmark",
    "select_device",
    "set_seed",
    "train_conditional_drifting",
]

_MODULE_EXPORTS = {
    "BenchmarkConfig": ("conditional_drifting.benchmark", "BenchmarkConfig"),
    "ConditionalDriftingGenerator": ("conditional_drifting.model", "ConditionalDriftingGenerator"),
    "DriftingConfig": ("conditional_drifting.training", "DriftingConfig"),
    "channel_registry": ("conditional_drifting.channels", "channel_registry"),
    "evaluate_residual_model": ("conditional_drifting.training", "evaluate_residual_model"),
    "run_single_channel_benchmark": ("conditional_drifting.benchmark", "run_single_channel_benchmark"),
    "select_device": ("conditional_drifting.training", "select_device"),
    "set_seed": ("conditional_drifting.training", "set_seed"),
    "train_conditional_drifting": ("conditional_drifting.training", "train_conditional_drifting"),
}


def __getattr__(name: str):
    if name == "baselines":
        return import_module("conditional_drifting.baselines")
    if name in _MODULE_EXPORTS:
        module_name, attr_name = _MODULE_EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
