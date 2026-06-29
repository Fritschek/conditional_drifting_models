"""Compact implementation of condition-wise Sinkhorn drifting."""

from .channels import get_channel
from .config import PAPER_CONFIG, VARIANTS
from .model import ConditionalGenerator
from .training import evaluate_generator, train_generator

__all__ = [
    "ConditionalGenerator",
    "PAPER_CONFIG",
    "VARIANTS",
    "evaluate_generator",
    "get_channel",
    "train_generator",
]

