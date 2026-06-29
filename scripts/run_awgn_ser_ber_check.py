import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import get_channel
from conditional_drifting.config import BUDGETS
from conditional_drifting.training import evaluate_generator, train_generator


class Encoder(nn.Module):
    def __init__(self, messages, code_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(messages, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, code_dim),
        )

    def forward(self, x):
        codes = self.net(x.float())
        return (codes - codes.mean()) / codes.std().clamp_min(1e-8)


class Decoder(nn.Module):
    def __init__(self, messages, code_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, messages),
        )

    def forward(self, y):
        return self.net(y.float())


def labels_to_one_hot(labels, messages):
    return F.one_hot(labels.long(), num_classes=messages).float()


def bit_error_rate(predictions, labels, messages):
    bits = max(1, int(math.ceil(math.log2(messages))))
    positions = torch.arange(bits, device=labels.device)
    pred_bits = (predictions[:, None] >> positions[None, :]) & 1
    label_bits = (labels[:, None] >> positions[None, :]) & 1
    return (pred_bits != label_bits).float().mean().item()


def evaluate_autoencoder(encoder, decoder, channel, cfg, args, device):
    encoder.eval()
    decoder.eval()
    total = 0
    symbol_errors = 0
    bit_errors = 0.0
    with torch.no_grad():
        remaining = args.eval_size
        while remaining > 0:
            batch = min(args.ae_batch_size, remaining)
            labels = torch.randint(0, args.messages, (batch,), device=device)
            x = labels_to_one_hot(labels, args.messages)
            codes = encoder(x)
            y = channel(codes, cfg["noise_std"])
            logits = decoder(y)
            predictions = logits.argmax(dim=1)
            symbol_errors += int((predictions != labels).sum().item())
            bit_errors += bit_error_rate(predictions, labels, args.messages) * batch
            total += batch
            remaining -= batch
    return {
        "ser": symbol_errors / max(1, total),
        "ber": bit_errors / max(1, total),
    }


def train_autoencoder(train_channel, eval_channel, cfg, args, device):
    encoder = Encoder(args.messages, cfg["n"], args.hidden_dim).to(device)
    decoder = Decoder(args.messages, cfg["n"], args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.ae_learning_rate)
    steps = max(1, args.ae_dataset_size // args.ae_batch_size)
    history = []
    for epoch in range(args.ae_epochs):
        losses = []
        for _ in range(steps):
            labels = torch.randint(0, args.messages, (args.ae_batch_size,), device=device)
            x = labels_to_one_hot(labels, args.messages)
            codes = encoder(x)
            y = train_channel(codes)
            logits = decoder(y)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = sum(losses) / len(losses)
        history.append(mean_loss)
        print("ae epoch %d/%d loss=%.6e" % (epoch + 1, args.ae_epochs, mean_loss), flush=True)
    metrics = evaluate_autoencoder(encoder, decoder, eval_channel, cfg, args, device)
    return metrics, history


def parse_args():
    parser = argparse.ArgumentParser(description="Quick AWGN SER/BER check for the compact drifting surrogate.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--messages", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--surrogate-epochs", type=int, default=None)
    parser.add_argument("--surrogate-budget", default="medium", choices=sorted(BUDGETS))
    parser.add_argument("--surrogate-dataset-size", type=int, default=None)
    parser.add_argument("--surrogate-batch-size", type=int, default=None)
    parser.add_argument("--surrogate-eval-size", type=int, default=None)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--ae-dataset-size", type=int, default=200000)
    parser.add_argument("--ae-batch-size", type=int, default=500)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-size", type=int, default=100000)
    parser.add_argument("--out", type=Path, default=Path("results/awgn_ser_ber_check.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    analytic = get_channel("AWGN")
    surrogate_overrides = {}
    if args.surrogate_epochs is not None:
        surrogate_overrides["epochs"] = args.surrogate_epochs
    if args.surrogate_dataset_size is not None:
        surrogate_overrides["dataset_size"] = args.surrogate_dataset_size
    if args.surrogate_batch_size is not None:
        surrogate_overrides["batch_size"] = args.surrogate_batch_size
    if args.surrogate_eval_size is not None:
        surrogate_overrides["eval_size"] = args.surrogate_eval_size
    surrogate, _, cfg = train_generator(
        analytic,
        "AWGN",
        "conditionwise_sinkhorn",
        seed=args.seed,
        device=device,
        budget=args.surrogate_budget,
        overrides=surrogate_overrides,
    )
    surrogate_metrics = evaluate_generator(
        surrogate,
        analytic,
        "AWGN",
        seed=args.seed + 1000,
        device=device,
        budget=args.surrogate_budget,
        overrides=surrogate_overrides,
    )

    def analytic_train_channel(codes):
        return analytic(codes, cfg["noise_std"])

    def surrogate_train_channel(codes):
        return surrogate(codes)

    print("[ser-check] analytic-channel AE", flush=True)
    analytic_metrics, analytic_history = train_autoencoder(analytic_train_channel, analytic, cfg, args, device)
    print("[ser-check] surrogate-channel AE", flush=True)
    surrogate_metrics_eval, surrogate_history = train_autoencoder(surrogate_train_channel, analytic, cfg, args, device)

    result = {
        "seed": args.seed,
        "surrogate_distribution_metrics": surrogate_metrics,
        "analytic_train_eval_analytic": analytic_metrics,
        "surrogate_train_eval_analytic": surrogate_metrics_eval,
        "analytic_loss": analytic_history,
        "surrogate_loss": surrogate_history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
