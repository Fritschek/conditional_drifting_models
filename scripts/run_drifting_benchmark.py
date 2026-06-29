import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import get_channel
from conditional_drifting.config import BUDGETS, VARIANTS
from conditional_drifting.training import evaluate_generator, train_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Train one compact drifting channel surrogate.")
    parser.add_argument("--channel", default="AWGN", choices=["AWGN", "Rayleigh", "SSPA", "TDL"])
    parser.add_argument("--variant", default="conditionwise_sinkhorn", choices=sorted(VARIANTS))
    parser.add_argument("--budget", default="default", choices=sorted(BUDGETS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/compact_run.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def selected_overrides(args):
    overrides = {}
    for name in ["epochs", "dataset_size", "eval_size", "batch_size"]:
        value = getattr(args, name.replace("-", "_"), None)
        if value is not None:
            overrides[name] = value
    return overrides


def main():
    args = parse_args()
    overrides = selected_overrides(args)
    channel = get_channel(args.channel)
    model, history, cfg = train_generator(
        channel,
        args.channel,
        args.variant,
        seed=args.seed,
        device=args.device,
        budget=args.budget,
        overrides=overrides,
    )
    metrics = evaluate_generator(
        model,
        channel,
        args.channel,
        seed=args.seed + 1000,
        device=args.device,
        budget=args.budget,
        overrides=overrides,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "channel": args.channel,
        "variant": args.variant,
        "seed": args.seed,
        "budget": args.budget,
        "config": cfg,
        "history": history,
        "metrics": metrics,
    }
    args.out.write_text(json.dumps(result, indent=2))
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "channel": args.channel}, args.checkpoint)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
