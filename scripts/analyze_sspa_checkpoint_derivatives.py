#!/usr/bin/env python3
"""Exploratory input-moment derivatives of completed SSPA checkpoints."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conditional_drifting.channels import sspa
from conditional_drifting.model import ConditionalDriftingGenerator
from conditional_drifting.sspa_checkpoint_metrics import output_jacobians, summarize_moments
from conditional_drifting.sspa_trajectory import source_hashes
from report_sspa_epsilon_trajectories import load_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchors", type=int, default=128)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=800201)
    args = parser.parse_args()
    if args.anchors < 1 or args.samples < 4 or args.samples % 2:
        parser.error("Positive anchors and an even sample count >=4 required")
    torch.set_num_threads(4)
    manifest, results, _, _ = load_suite(args.suite_dir)
    if args.out_dir.exists():
        raise FileExistsError("Use a new analysis directory")
    args.out_dir.mkdir(parents=True)
    rows, checkpoint_hashes = [], {}
    for (seed, policy), result in results.items():
        c = result["config"]
        torch.manual_seed(args.seed)
        # Generate panels on CPU for reproducible inputs across analysis devices.
        anchors = torch.randn(args.anchors, c["n"]).to(args.device)
        latent = torch.randn(args.anchors * args.samples, c["latent_dim"]).to(args.device)
        truth_inputs = anchors.detach().clone().requires_grad_(True)
        truth = sspa(truth_inputs, 0., args.device)
        truth_jacobian = output_jacobians(truth, truth_inputs).detach()
        truth = truth.detach()
        model = ConditionalDriftingGenerator(c["n"], c["n"], c["latent_dim"], c["hidden_dim"]).to(args.device).eval()
        for update in manifest["checkpoints"]:
            path = args.suite_dir / policy / f"seed{seed}" / f"update{update}.pt"
            checkpoint_hashes[str(path.relative_to(args.suite_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["config"] != c or state["update"] != update or state["source_hashes"] != source_hashes():
                raise ValueError(f"Checkpoint provenance mismatch: {path}")
            model.load_state_dict(state["model"])
            inputs = anchors.repeat_interleave(args.samples, dim=0).requires_grad_(True)
            generated = model(inputs, latent)
            jacobians = output_jacobians(generated, inputs).detach().reshape(args.anchors, args.samples, c["n"], c["n"])
            generated = generated.detach().reshape(args.anchors, args.samples, c["n"])
            with torch.no_grad():
                weights = {name: value.detach().cpu().double() for name, value in model.named_parameters()}
                weight_metrics = {"parameter_l2": sum(v.square().sum() for v in weights.values()).sqrt().item()}
                for name, value in weights.items():
                    if value.ndim == 2:
                        weight_metrics[name + "_fro"] = value.norm().item()
                        weight_metrics[name + "_operator"] = torch.linalg.matrix_norm(value, ord=2).item()
                for label, sl in (("all", slice(None)), ("half_a", slice(0, args.samples // 2)),
                                  ("half_b", slice(args.samples // 2, None))):
                    row = dict(seed=seed, policy=policy, update=update, panel=label, **weight_metrics,
                               **summarize_moments(generated[:, sl], jacobians[:, sl], truth, truth_jacobian, c["noise_std"]))
                    rows.append(row)
            print(json.dumps({"seed": seed, "policy": policy, "update": update}), flush=True)
    with (args.out_dir / "derivatives.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(suite_dir=str(args.suite_dir.resolve()), suite_manifest_sha256=hashlib.sha256(
        (args.suite_dir / "manifest.json").read_bytes()).hexdigest(), checkpoints=checkpoint_hashes,
        analysis_seed=args.seed, anchors=args.anchors, samples=args.samples, device=args.device,
        torch_version=str(torch.__version__), exploratory=True,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).resolve().parents[1] / "conditional_drifting/sspa_checkpoint_metrics.py")})
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    definitions = (
        ("mean_error_l2", "Conditional mean error"),
        ("covariance_error_fro", "Conditional covariance error"),
        ("mean_jacobian_relative_error", "Mean Jacobian relative error"),
        ("covariance_jacobian_rms_fro", "Covariance derivative norm (true: zero)"),
        ("parameter_l2", "Total parameter norm"),
        ("sample_jacobian_rms_fro", "Sample input-Jacobian norm"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    for policy, color in zip(manifest["policies"], ("#0072B2", "#D55E00", "#009E73")):
        for j, seed in enumerate(manifest["seeds"]):
            selected = [r for r in rows if r["policy"] == policy and r["seed"] == seed
                        and r["panel"] == "all" and r["update"] >= 30000]
            for ax, (metric, _) in zip(axes.flat, definitions):
                ax.plot([r["update"] / 1000 for r in selected], [r[metric] for r in selected],
                        color=color, linestyle=["-", "--", ":"][j % 3], marker="o", markersize=3,
                        label=policy if j == 0 else None)
    for ax, (_, title) in zip(axes.flat, definitions):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training updates (thousands)")
        ax.set_yscale("log")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.out_dir / "checkpoint_derivatives.pdf")
    fig.savefig(args.out_dir / "checkpoint_derivatives.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
