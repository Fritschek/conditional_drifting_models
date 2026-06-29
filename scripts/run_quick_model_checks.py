import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import get_channel
from conditional_drifting.config import BUDGETS, CHANNEL_CONFIG, VARIANTS
from conditional_drifting.training import evaluate_generator, train_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Run quick checks for every compact drifting variant and paper channel.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", default="AWGN,Rayleigh,SSPA,TDL")
    parser.add_argument("--variants", default="direct,joint_sinkhorn,conditionwise_sinkhorn")
    parser.add_argument("--budget", default="quick", choices=sorted(BUDGETS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--eval-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/quick_model_checks.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    channels = [name.strip() for name in args.channels.split(",") if name.strip()]
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown_channels = [name for name in channels if name not in CHANNEL_CONFIG]
    unknown_variants = [name for name in variants if name not in VARIANTS]
    if unknown_channels:
        raise ValueError("Unknown channels: %s" % ", ".join(unknown_channels))
    if unknown_variants:
        raise ValueError("Unknown variants: %s" % ", ".join(unknown_variants))

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.dataset_size is not None:
        overrides["dataset_size"] = args.dataset_size
    if args.eval_size is not None:
        overrides["eval_size"] = args.eval_size
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    rows = []
    for channel_name in channels:
        channel = get_channel(channel_name)
        for variant_name in variants:
            print("[quick-check] channel=%s variant=%s" % (channel_name, variant_name), flush=True)
            model, history, cfg = train_generator(
                channel,
                channel_name,
                variant_name,
                seed=args.seed,
                device=args.device,
                budget=args.budget,
                overrides=overrides,
            )
            metrics = evaluate_generator(
                model,
                channel,
                channel_name,
                seed=args.seed + 1000,
                device=args.device,
                budget=args.budget,
                overrides=overrides,
            )
            rows.append(
                {
                    "channel": channel_name,
                    "variant": variant_name,
                    "final_loss": history[-1]["loss"],
                    "final_drift_norm": history[-1]["drift_norm"],
                    "direct_swd": metrics["direct_swd"],
                    "anchor_swd": metrics["anchor_swd"],
                    "n": cfg["n"],
                    "noise_std": cfg["noise_std"],
                }
            )

    result = {
        "seed": args.seed,
        "budget": args.budget,
        "overrides": overrides,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
