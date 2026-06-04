from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import channel_registry
from conditional_drifting.e2e_implants import load_implant_from_checkpoint
from conditional_drifting.paper2309_presets import PAPER2309_PRESETS, ebno_to_noise
from conditional_drifting.training import select_device, set_seed


DEFAULT_LABELS = {
    "kernel_target": "Target kernel",
    "kernel_joint": "Joint kernel",
    "joint_sinkhorn": "Joint Sinkhorn",
    "fiber_sinkhorn": "Condition-wise Sinkhorn",
    "fiber_sinkhorn_marginal": "Condition-wise Sinkhorn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fixed-condition output clouds for analytic and learned channel implants.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channel", type=str, default="SSPA", choices=sorted(PAPER2309_PRESETS))
    parser.add_argument("--wflow-suite-dir", type=Path, required=True)
    parser.add_argument(
        "--variant-suite-map",
        type=str,
        default="",
        help="Optional comma-separated map VARIANT=/path for variants stored outside --wflow-suite-dir.",
    )
    parser.add_argument("--variants", type=str, default="joint_sinkhorn,fiber_sinkhorn")
    parser.add_argument("--variant-labels", type=str, default="", help="Optional comma-separated map VARIANT=Label.")
    parser.add_argument("--num-anchors", type=int, default=3)
    parser.add_argument("--candidate-anchors", type=int, default=256)
    parser.add_argument("--samples-per-anchor", type=int, default=512)
    parser.add_argument("--dims", type=str, default="0,1")
    parser.add_argument("--out-dir", type=Path, default=Path("results/conditional_fiber_diagnostics"))
    parser.add_argument("--figure-pdf", type=Path, default=Path("Journal_version/figures/conditional_fiber_diagnostic_sspa.pdf"))
    parser.add_argument("--figure-png", type=Path, default=Path("Journal_version/figures/conditional_fiber_diagnostic_sspa.png"))
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_map(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in parse_csv_list(text):
        if "=" not in item:
            raise ValueError(f"Invalid mapping item {item!r}; expected KEY=VALUE.")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def checkpoint_path(args: argparse.Namespace, variant: str) -> Path:
    suite_map = {key: Path(value) for key, value in parse_map(args.variant_suite_map).items()}
    suite_dir = suite_map.get(variant, args.wflow_suite_dir)
    return (
        suite_dir
        / variant
        / f"seed{args.seed}"
        / "checkpoints"
        / f"enhanced_direct_{args.channel.lower()}_seed{args.seed}.pt"
    )


def select_anchor_indices(candidates: torch.Tensor, num_anchors: int, dims: tuple[int, int]) -> torch.Tensor:
    if num_anchors >= candidates.shape[0]:
        return torch.arange(candidates.shape[0], device=candidates.device)
    xy = candidates[:, list(dims)]
    angles = torch.atan2(xy[:, 1], xy[:, 0])
    radii = torch.linalg.norm(xy, dim=1)
    selected: list[int] = []
    target_angles = torch.linspace(-math.pi, math.pi, steps=num_anchors + 1, device=candidates.device)[:-1]
    for target in target_angles:
        angle_error = torch.atan2(torch.sin(angles - target), torch.cos(angles - target)).abs()
        score = angle_error - 0.15 * radii
        for idx in torch.argsort(score):
            item = int(idx.item())
            if item not in selected:
                selected.append(item)
                break
    return torch.as_tensor(selected, device=candidates.device, dtype=torch.long)


def covariance_fro(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[0] < 2 or right.shape[0] < 2:
        return float("nan")
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    cov_left = left_centered.T @ left_centered / max(1, left.shape[0] - 1)
    cov_right = right_centered.T @ right_centered / max(1, right.shape[0] - 1)
    return float(torch.linalg.norm(cov_left - cov_right, ord="fro").detach().cpu().item())


def sample_clouds(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    device = select_device(args.device)
    set_seed(args.seed)
    preset = PAPER2309_PRESETS[args.channel]
    channel_fn = channel_registry()[args.channel]
    noise_std = ebno_to_noise(float(preset.ebn0_db), float(preset.rate))
    dims_list = [int(item) for item in parse_csv_list(args.dims)]
    if len(dims_list) != 2:
        raise ValueError("--dims must contain exactly two comma-separated dimensions.")
    dims = (dims_list[0], dims_list[1])

    candidates = torch.randn(int(args.candidate_anchors), int(preset.n), device=device)
    anchor_idx = select_anchor_indices(candidates, int(args.num_anchors), dims)
    anchors = candidates.index_select(0, anchor_idx)

    variants = parse_csv_list(args.variants)
    labels = {**DEFAULT_LABELS, **parse_map(args.variant_labels)}
    implants = {}
    for variant in variants:
        ckpt = checkpoint_path(args, variant)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for {variant}: {ckpt}")
        implants[variant] = load_implant_from_checkpoint(ckpt, device=device)

    clouds: dict[str, list[torch.Tensor]] = {"analytic": []}
    for variant in variants:
        clouds[variant] = []

    rows: list[dict[str, object]] = []
    for anchor_id, anchor in enumerate(anchors):
        x_rep = anchor.unsqueeze(0).repeat(int(args.samples_per_anchor), 1)
        y_true = channel_fn(x_rep, noise_std, device)
        clouds["analytic"].append(y_true.detach().cpu())
        for variant, implant in implants.items():
            y_pred = implant(x_rep, ebno_db=float(preset.ebn0_db), rate=float(preset.rate), device=device)
            clouds[variant].append(y_pred.detach().cpu())
            rows.append(
                {
                    "anchor": anchor_id,
                    "variant": variant,
                    "label": labels.get(variant, variant),
                    "mean_l2": float(torch.linalg.norm(y_true.mean(dim=0) - y_pred.mean(dim=0)).detach().cpu().item()),
                    "cov_fro": covariance_fro(y_true, y_pred),
                    "anchor_norm": float(torch.linalg.norm(anchor).detach().cpu().item()),
                    "anchor_dim0": float(anchor[dims[0]].detach().cpu().item()),
                    "anchor_dim1": float(anchor[dims[1]].detach().cpu().item()),
                }
            )

    payload = {
        "seed": int(args.seed),
        "channel": args.channel,
        "ebno_db": float(preset.ebn0_db),
        "rate": float(preset.rate),
        "noise_std": float(noise_std),
        "dims": list(dims),
        "num_anchors": int(args.num_anchors),
        "samples_per_anchor": int(args.samples_per_anchor),
        "variants": variants,
        "labels": labels,
        "anchors": anchors.detach().cpu().tolist(),
        "clouds": clouds,
    }
    return payload, rows


def write_metrics(out_dir: Path, rows: list[dict[str, object]], payload: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "conditional_fiber_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor", "variant", "label", "mean_l2", "cov_fro", "anchor_norm", "anchor_dim0", "anchor_dim1"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: value for key, value in payload.items() if key != "clouds"}
    summary["metrics_csv"] = str(csv_path)
    (out_dir / "conditional_fiber_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot(payload: dict[str, object], rows: list[dict[str, object]], figure_pdf: Path, figure_png: Path) -> None:
    import matplotlib.pyplot as plt

    variants = list(payload["variants"])
    methods = ["analytic", *variants]
    labels = dict(payload["labels"])
    dims = tuple(int(v) for v in payload["dims"])
    clouds = payload["clouds"]
    num_anchors = int(payload["num_anchors"])
    colors = {
        "analytic": "#4d4d4d",
        "joint_sinkhorn": "#e15759",
        "kernel_joint": "#f28e2b",
        "kernel_target": "#59a14f",
        "fiber_sinkhorn": "#4e79a7",
        "fiber_sinkhorn_marginal": "#4e79a7",
    }

    fig, axes = plt.subplots(
        num_anchors,
        len(methods),
        figsize=(2.55 * len(methods), 2.15 * num_anchors),
        squeeze=False,
        constrained_layout=True,
    )
    for anchor_id in range(num_anchors):
        analytic = clouds["analytic"][anchor_id]
        all_xy = [analytic[:, list(dims)]]
        for variant in variants:
            all_xy.append(clouds[variant][anchor_id][:, list(dims)])
        stacked = torch.cat(all_xy, dim=0)
        pad = 0.08 * (stacked.max(dim=0).values - stacked.min(dim=0).values).clamp_min(1e-3)
        xlim = (float(stacked[:, 0].min() - pad[0]), float(stacked[:, 0].max() + pad[0]))
        ylim = (float(stacked[:, 1].min() - pad[1]), float(stacked[:, 1].max() + pad[1]))

        for col, method in enumerate(methods):
            ax = axes[anchor_id][col]
            if method == "analytic":
                data = analytic
                ax.scatter(data[:, dims[0]], data[:, dims[1]], s=7, alpha=0.36, color=colors["analytic"], linewidths=0)
            else:
                ax.scatter(
                    analytic[:, dims[0]],
                    analytic[:, dims[1]],
                    s=7,
                    alpha=0.16,
                    color="#8a8a8a",
                    linewidths=0,
                    label="analytic",
                )
                data = clouds[method][anchor_id]
                ax.scatter(data[:, dims[0]], data[:, dims[1]], s=7, alpha=0.46, color=colors.get(method, "#4e79a7"), linewidths=0)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.3, alpha=0.25)
            if anchor_id == 0:
                ax.set_title("Analytic" if method == "analytic" else labels.get(method, method), fontsize=8)
            if col == 0:
                ax.set_ylabel(f"anchor {anchor_id + 1}\n$y_{{{dims[1] + 1}}}$", fontsize=8)
            else:
                ax.set_yticklabels([])
            if anchor_id == num_anchors - 1:
                ax.set_xlabel(f"$y_{{{dims[0] + 1}}}$", fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="both", labelsize=7, length=2)

    figure_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_pdf, bbox_inches="tight")
    fig.savefig(figure_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payload, rows = sample_clouds(args)
    write_metrics(args.out_dir, rows, payload)
    plot(payload, rows, args.figure_pdf, args.figure_png)
    print(
        json.dumps(
            {
                "summary": str(args.out_dir / "conditional_fiber_summary.json"),
                "csv": str(args.out_dir / "conditional_fiber_metrics.csv"),
                "figure_pdf": str(args.figure_pdf),
                "figure_png": str(args.figure_png),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
