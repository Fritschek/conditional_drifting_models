from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import channel_registry
from conditional_drifting.symbolic_ae import (
    SymbolicDecoder,
    SymbolicEncoder,
    labels_to_one_hot,
    sample_message_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an AWGN SER/BER figure from saved symbolic autoencoder checkpoints.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--ebno-values", type=str, default="0,1,2,3,4,5,6,7,8")
    parser.add_argument(
        "--out-stem",
        type=str,
        default=str(ROOT / "Paper_camera_ready" / "figures" / "awgn_ser_ber_symbolic"),
    )
    parser.add_argument("--direct-checkpoint", type=str, default=None, help="Optional override for the direct drifting checkpoint.")
    parser.add_argument("--direct-label", type=str, default="Drifting (dir.)", help="Legend label for the primary direct drifting curve.")
    parser.add_argument("--extra-direct-checkpoint", type=str, default=None, help="Optional second direct drifting checkpoint.")
    parser.add_argument("--extra-direct-label", type=str, default="Drifting (dir., cond. kernel)", help="Legend label for the extra direct drifting curve.")
    parser.add_argument("--residual-checkpoint", type=str, default=None, help="Optional override for the residual drifting checkpoint.")
    parser.add_argument("--include-residual", action="store_true", help="Also plot the strongest saved residual-drifting checkpoint.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def ebno_to_noise(ebno_db: float, rate: float) -> float:
    ebno_linear = 10.0 ** (float(ebno_db) / 10.0)
    return math.sqrt(1.0 / (2.0 * float(rate) * ebno_linear))


def load_symbolic_checkpoint(path: Path, device: torch.device) -> tuple[SymbolicEncoder, SymbolicDecoder, dict]:
    payload = torch.load(path, map_location=device)
    cfg = payload["config"]
    encoder = SymbolicEncoder(
        message_dim=int(cfg["message_dim"]),
        code_dim=int(cfg["code_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    decoder = SymbolicDecoder(
        message_dim=int(cfg["message_dim"]),
        code_dim=int(cfg["code_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
    ).to(device)
    encoder.load_state_dict(payload["encoder_state"])
    decoder.load_state_dict(payload["decoder_state"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, payload


def compute_bit_error_rate(predictions: torch.Tensor, labels: torch.Tensor, *, bits_per_symbol: int) -> float:
    shifts = torch.arange(bits_per_symbol, device=predictions.device)
    pred_bits = ((predictions.unsqueeze(1) >> shifts) & 1).float()
    true_bits = ((labels.unsqueeze(1) >> shifts) & 1).float()
    return float((pred_bits != true_bits).float().mean().item())


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    channel_name: str,
    ebno_db: float,
    eval_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    torch.manual_seed(seed)
    encoder, decoder, payload = load_symbolic_checkpoint(checkpoint_path, device)
    cfg = payload["config"]
    message_dim = int(cfg["message_dim"])
    bits_per_symbol = int(round(math.log2(message_dim)))
    rate = 4.0 / 7.0
    noise_std = ebno_to_noise(ebno_db, rate)
    channel_fn = channel_registry()[channel_name]
    steps = max(1, math.ceil(eval_size / batch_size))
    ser_total = 0.0
    ber_total = 0.0

    for _ in range(steps):
        labels = sample_message_labels(batch_size, message_dim, device)
        messages = labels_to_one_hot(labels, message_dim)
        encoded = encoder(messages)
        received = channel_fn(encoded, noise_std, device)
        logits = decoder(received)
        preds = torch.argmax(logits, dim=1)
        ser_total += float((preds != labels).float().mean().item())
        ber_total += compute_bit_error_rate(preds, labels, bits_per_symbol=bits_per_symbol)

    return {
        "ser": ser_total / steps,
        "ber": ber_total / steps,
    }


def default_curves(
    include_residual: bool,
    *,
    direct_checkpoint: str | None,
    direct_label: str,
    extra_direct_checkpoint: str | None,
    extra_direct_label: str,
    residual_checkpoint: str | None,
) -> list[dict[str, object]]:
    curves = [
        {
            "label": "Analytic train",
            "path": ROOT / "weights" / "symbolic_autoencoders" / "awgn" / "symbolic_awgn_analytic_seed7.pt",
            "color": "#4c72b0",
            "linestyle": "-",
            "marker": "o",
        },
        {
            "label": direct_label,
            "path": Path(direct_checkpoint) if direct_checkpoint else ROOT / "weights" / "symbolic_autoencoders" / "awgn" / "symbolic_awgn_drifting_direct_highbudget_queue_seed7.pt",
            "color": "#dd8452",
            "linestyle": "-",
            "marker": "s",
        },
        {
            "label": "Diffusion (dir.)",
            "path": ROOT / "weights" / "symbolic_autoencoders" / "awgn" / "symbolic_awgn_diffusion_direct_seed7.pt",
            "color": "#55a868",
            "linestyle": "-",
            "marker": "^",
        },
        {
            "label": "Paper WGAN",
            "path": ROOT / "weights" / "symbolic_autoencoders" / "awgn" / "symbolic_awgn_wgan_seed7.pt",
            "color": "#c44e52",
            "linestyle": "-",
            "marker": "D",
        },
    ]
    if extra_direct_checkpoint:
        curves.insert(
            2,
            {
                "label": extra_direct_label,
                "path": Path(extra_direct_checkpoint),
                "color": "#937860",
                "linestyle": "-",
                "marker": "P",
            },
        )
    if include_residual:
        curves.append(
            {
                "label": "Drifting (res.)",
                "path": Path(residual_checkpoint) if residual_checkpoint else ROOT / "weights" / "symbolic_autoencoders" / "awgn" / "symbolic_awgn_drifting_residual_queue_seed7.pt",
                "color": "#8172b3",
                "linestyle": "--",
                "marker": "v",
            }
        )
    return curves


def configure_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="deep",
        font="DejaVu Serif",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fbfbfc",
            "grid.color": "#d8dbe2",
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
            "axes.edgecolor": "#111827",
            "axes.linewidth": 0.9,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ebno_values = [float(part.strip()) for part in args.ebno_values.split(",") if part.strip()]
    out_stem = Path(args.out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    curves = default_curves(
        args.include_residual,
        direct_checkpoint=args.direct_checkpoint,
        direct_label=args.direct_label,
        extra_direct_checkpoint=args.extra_direct_checkpoint,
        extra_direct_label=args.extra_direct_label,
        residual_checkpoint=args.residual_checkpoint,
    )

    rows: list[dict[str, object]] = []
    for curve in curves:
        checkpoint_path = Path(curve["path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        for ebno_db in ebno_values:
            metrics = evaluate_checkpoint(
                checkpoint_path,
                channel_name="AWGN",
                ebno_db=ebno_db,
                eval_size=args.eval_size,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
            )
            rows.append(
                {
                    "label": curve["label"],
                    "checkpoint": str(checkpoint_path),
                    "ebno_db": ebno_db,
                    "ser": metrics["ser"],
                    "ber": metrics["ber"],
                }
            )
            print(
                f"{curve['label']}: Eb/N0={ebno_db:.1f} dB, "
                f"SER={metrics['ser']:.6e}, BER={metrics['ber']:.6e}",
                flush=True,
            )

    csv_path = out_stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "checkpoint", "ebno_db", "ser", "ber"])
        writer.writeheader()
        writer.writerows(rows)

    configure_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.0), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.17, top=0.93)
    for curve in curves:
        pts = [row for row in rows if row["label"] == curve["label"]]
        xs = [float(row["ebno_db"]) for row in pts]
        ys = [float(row["ser"]) for row in pts]
        ax.semilogy(
            xs,
            ys,
            marker=str(curve["marker"]),
            ms=4,
            lw=2.2,
            label=str(curve["label"]),
            color=str(curve["color"]),
            linestyle=str(curve["linestyle"]),
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
    ax.grid(True, which="both")
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("SER")
    ax.set_title("AWGN")
    ax.set_xlim(min(ebno_values), max(ebno_values))
    ax.set_ylim(1e-5, 3e-1)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.minorticks_on()
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#111827")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis="both", which="major", length=5.0, width=0.95, direction="out", colors="#111827", bottom=True, left=True)
    ax.tick_params(axis="x", which="minor", length=1.6, width=0.45, direction="out", colors="#6b7280", bottom=True)
    ax.tick_params(axis="y", which="minor", length=1.6, width=0.45, direction="out", colors="#6b7280", left=True)
    ax.legend(loc="lower left", frameon=False)

    pdf_path = out_stem.with_suffix(".pdf")
    png_path = out_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"figure_pdf={pdf_path}")
    print(f"figure_png={png_path}")
    print(f"figure_csv={csv_path}")


if __name__ == "__main__":
    main()
