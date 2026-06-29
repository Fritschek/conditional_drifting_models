"""Paper defaults collected in one place."""

import math


def ebno_to_noise(ebn0_db, rate):
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 1.0 / math.sqrt(2.0 * rate * ebn0)

PAPER_CONFIG = {
    "dataset_size": 120000,
    "batch_size": 512,
    "epochs": 30,
    "learning_rate": 1e-3,
    "hidden_dim": 128,
    "latent_dim": 16,
    "eval_size": 100000,
    "swd_projections": 128,
    "kernel_min_bandwidth": 0.2,
    "max_drift_norm": 2.0,
    "sinkhorn_min_epsilon": 1e-3,
    "sinkhorn_iterations": 10,
    "fiber_generated_samples": 4,
    "fiber_positive_samples": 4,
}

CHANNEL_CONFIG = {
    "AWGN": {
        "n": 7,
        "noise_std": ebno_to_noise(5.0, 4.0 / 7.0),
    },
    "Rayleigh": {
        "n": 7,
        "noise_std": ebno_to_noise(12.0, 4.0 / 7.0),
    },
    "SSPA": {
        "n": 8,
        "noise_std": ebno_to_noise(8.0, 6.0 / 8.0),
    },
    "TDL": {
        "n": 8,
        "noise_std": ebno_to_noise(10.0, 4.0 / 8.0),
    },
}

BUDGETS = {
    "default": {
        "all": {},
    },
    "quick": {
        "all": {
            "epochs": 2,
            "dataset_size": 2048,
            "eval_size": 2048,
            "batch_size": 128,
            "swd_projections": 64,
        },
    },
    "medium": {
        "all": {
            "epochs": 30,
            "dataset_size": 1000000,
            "eval_size": 200000,
            "batch_size": 5000,
            "swd_projections": 128,
        },
    },
    "paper": {
        "AWGN": {
            "epochs": 30,
            "dataset_size": 10000000,
            "eval_size": 10000000,
            "batch_size": 5000,
            "swd_projections": 128,
        },
        "Rayleigh": {
            "epochs": 30,
            "dataset_size": 10000000,
            "eval_size": 10000000,
            "batch_size": 5000,
            "swd_projections": 128,
        },
        "SSPA": {
            "epochs": 160,
            "dataset_size": 10000000,
            "eval_size": 10000000,
            "batch_size": 4096,
            "swd_projections": 128,
        },
        "TDL": {
            "epochs": 60,
            "dataset_size": 120000,
            "eval_size": 100000,
            "batch_size": 512,
            "swd_projections": 128,
        },
    },
}


def build_config(channel_name, budget="default", overrides=None):
    if budget not in BUDGETS:
        raise ValueError("Unknown budget %r. Available: %s" % (budget, ", ".join(BUDGETS)))
    cfg = dict(PAPER_CONFIG)
    cfg.update(CHANNEL_CONFIG[channel_name])
    cfg.update(BUDGETS[budget].get("all", {}))
    cfg.update(BUDGETS[budget].get(channel_name, {}))
    if overrides:
        cfg.update(overrides)
    return cfg


VARIANTS = {
    "direct": {
        "label": "Direct drifting",
        "loss": "kernel",
        "joint_conditioning": False,
    },
    "joint_sinkhorn": {
        "label": "Joint Sinkhorn drifting",
        "loss": "sinkhorn",
        "joint_conditioning": True,
    },
    "conditionwise_sinkhorn": {
        "label": "Condition-wise Sinkhorn drifting",
        "loss": "conditionwise_sinkhorn",
        "joint_conditioning": False,
    },
}
