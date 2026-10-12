"""Export every completed application-pilot row, with counts and measured costs."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/conditional-drifting-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = dict(analytic="Analytic", gaussian="Gaussian", wgan="WGAN",
    diffusion="DDIM-20", flow="Flow-8", sinkhorn="Sinkhorn K64",
    consistency_distilled="Consistency distillation")
COLORS = dict(zip(LABELS, ("#333333", "#a07800", "#008f87", "#2864bc", "#b95680", "#c3452b", "#7744a4")))


def number(value):
    return "unresolved" if value is None else f"{value:.5g}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--codebook-check", type=Path)
    p.add_argument("--support-check", type=Path)
    args = p.parse_args()
    data = json.loads((args.suite_dir / "results.json").read_text())
    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    rows = data["rows"]
    if len(rows) != len(LABELS) or {r["method"] for r in rows} != set(LABELS):
        raise ValueError("Expected analytic plus all six learned methods")
    budget = manifest["settings"]["updates"] * manifest["settings"]["batch_size"]
    trials = manifest["settings"]["ae_eval_samples"] * manifest["channel_spec"]["message_dim"]
    curves = []
    for row in rows:
        if row["method"] != "analytic" and row["cost"]["oracle_outputs"] != budget:
            raise ValueError("Mismatched oracle count")
        for point in row["analytic_channel_evaluation"]:
            if point["trials"] != trials or abs(point["ser"]-point["errors"]/trials) > 1e-12:
                raise ValueError("SER/trial accounting mismatch")
            curves.append(dict(method=row["method"], **point))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "curves.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    lines = [f"# {data['channel']} fast-generator application pilot", "",
        f"Development seed {data['seed']}. Fixed final EMA checkpoints, no downstream selection.",
        f"Each learned method receives {budget:,} analytic training outputs. This is not matched compute.",
        "The consistency model includes learned EDM teacher pretraining. All downstream evaluation",
        "uses the analytic channel, after fresh AE training through each frozen surrogate.", "",
        "## Communication results", "",
        "| Method | Eb/N0 (dB) | SER | Symbol errors / trials | Natural-label BER | Bit errors |",
        "|---|---:|---:|---|---:|---:|"]
    for r in curves:
        lines.append(f"| {LABELS[r['method']]} | {r['ebno_db']:g} | {r['ser']:.6g} | "
            f"{r['errors']} / {r['trials']} | {r['ber']:.6g} | {r['bit_errors']} |")
    lines += ["", "## Distribution and moments", "",
        "Conditional metrics are output-space distances at fixed inputs. Floors use two independent",
        "analytic output clouds. Codebook rows use the final common analytic-trained encoder.", "",
        "| Method | Global SWD | Global floor | Anchor SWD / floor | Codebook SWD / floor | Mean RMSE | Variance / truth |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows[1:]:
        d, c, m = r["distribution"], r["codebook_distributions"]["final"], r["moments"]
        lines.append(f"| {LABELS[r['method']]} | {d['global_swd']:.5g} | {d['global_floor']:.5g} | "
            f"{d['conditional']['swd']/d['analytic_floor']['swd']:.5g} | "
            f"{c['conditional']['swd']/c['analytic_floor']['swd']:.5g} | {m['mean_rmse']:.5g} | {m['variance_ratio']:.5g} |")
    lines += ["", "## Common-codec gradients", "",
        "Mean-vector comparisons include every method. Interpret direction comparisons only when",
        "the analytic reference is resolved; the flag is a Monte Carlo heuristic, not a guarantee.", "",
        "| Method | Codec | Space | Cosine | Relative error | Norm ratio | Ref signal / MC SE | Split cosine | Resolved |",
        "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows[1:]:
        for label, spaces in r["gradients"].items():
            for space in ("code", "encoder"):
                g = spaces[space]
                a = g["alignment"]
                lines.append(f"| {LABELS[r['method']]} | {label} | {space} | {number(a['cosine'])} | "
                    f"{number(a['relative_error'])} | {number(a['norm_ratio'])} | "
                    f"{number(g['reference_signal_to_mc_se'])} | {number(g['reference_split_alignment']['cosine'])} | {g['reference_resolved']} |")
    lines += ["", "## Measured costs", "",
        "Seconds cover synchronized training updates, including optimizer/EMA work, excluding",
        "initialization, evaluation and checkpoint I/O. No extrapolation. Latency is sampling only.", "",
        "| Method | Parameters | NFE | Generator train (s) | AE train (s) | Batch-1 (ms) | Batch-500 amortized (ms/sample) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    batch = manifest["settings"]["ae_batch"]
    lines[-2] = lines[-2].replace("Batch-500", f"Batch-{batch}")
    for r in rows:
        if r["method"] == "analytic":
            lines.append(f"| Analytic | - | - | - | {r['ae_train_seconds']:.2f} | - | - |")
            continue
        cost = r["cost"]
        t = {v["batch"]: v for v in r["timing"] if v["operation"] == "sampling"}
        lines.append(f"| {LABELS[r['method']]} | {cost['inference_parameters']} | {cost['nfe']} | "
            f"{cost['train_seconds']:.2f} | {r['ae_train_seconds']:.2f} | {t[1]['mean_ms']:.5g} | {t[batch]['amortized_ms_per_sample']:.5g} |")
    if args.codebook_check is not None:
        checks = json.loads(args.codebook_check.read_text())
        digest = hashlib.sha256((args.suite_dir / "results.json").read_bytes()).hexdigest()
        if checks["source_results_sha256"] != digest:
            raise ValueError("Codebook check belongs to a different result snapshot")
        lines += ["", "## Own-codebook transfer check (post hoc)", "",
            "The same final surrogate-trained codec is evaluated through both channels at nominal SNR.",
            "These checks did not affect training or selection. Common-codec and own-codebook metrics",
            "answer different questions; they should not be pooled.", "",
            "| Method | Own anchor SWD / floor | Mean RMSE | Variance / truth | Surrogate SER | Analytic SER | Analytic errors / trials |",
            "|---|---:|---:|---:|---:|---:|---|"]
        for r in checks["rows"]:
            d, a, s = r["distribution"], r["analytic"], r["surrogate"]
            lines.append(f"| {LABELS[r['method']]} | {d['conditional']['swd']/d['analytic_floor']['swd']:.5g} | "
                f"{r['mean_rmse']:.5g} | {r['variance_ratio']:.5g} | {s['ser']:.6g} | {a['ser']:.6g} | "
                f"{a['errors']} / {a['trials']} |")
    if args.support_check is not None:
        support = json.loads(args.support_check.read_text())
        if args.codebook_check is None or support["own_codebook_check_sha256"] != hashlib.sha256(args.codebook_check.read_bytes()).hexdigest():
            raise ValueError("Input-coverage check needs its matching codebook evaluation")
        lines += ["", "## Input coverage and error localization (post hoc)", "",
            "Every codebook has mean squared norm eight. The power constraint does not bound each",
            "individual codeword. Training uses iid standard-normal coordinates; the radial tail",
            "is computed from chi-square(8). Error counts exactly replay the preceding evaluation.", "",
            "| Method | Maximum input norm | Training-law probability above that norm | Errors at two largest-norm words / all errors |",
            "|---|---:|---:|---|"]
        for r in support["rows"]:
            ids = sorted(range(len(r["input_norms"])), key=lambda i: r["input_norms"][i], reverse=True)[:2]
            errors = r["analytic_errors_per_message"]
            lines.append(f"| {LABELS[r['method']]} | {r['input_norms'][ids[0]]:.5g} | "
                f"{r['radial_tail_probabilities'][ids[0]]:.5g} | {sum(errors[i] for i in ids)} / {sum(errors)} |")
    lines += ["", "## Scope", "", "One development seed does not establish a significant model-family ranking.",
        "The oracle observations are equal but distinct-input coverage and training compute differ.",
        "The analytic SSPA conditional law is Gaussian; this channel alone cannot establish a need",
        "for a flexible non-Gaussian generator. Fixed NFEs and recipes are not per-family optima.",
        "BER uses natural binary labels; there is no claim of optimized bit labeling.", "",
        f"Source suite: `{args.suite_dir}`. See the frozen protocol for all budgets and settings."]
    (args.out_dir / "results.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for r in rows:
        points = r["analytic_channel_evaluation"]
        for ax, metric in zip(axes, ("ser", "ber")):
            # Zero counts are left as gaps on logarithmic axes, not invented error rates.
            ax.plot([v["ebno_db"] for v in points], [v[metric] if v[metric] > 0 else float("nan") for v in points],
                label=LABELS[r["method"]], color=COLORS[r["method"]], marker="o", markersize=4,
                linestyle="--" if r["method"] == "analytic" else "-")
    for ax, metric in zip(axes, ("SER", "Natural-label BER")):
        ax.set_xlabel("Evaluation Eb/N0 (dB)")
        ax.set_ylabel(metric)
        ax.set_yscale("log")
        ax.set_xticks([v["ebno_db"] for v in rows[0]["analytic_channel_evaluation"]])
        ax.grid(alpha=.2, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{data['channel']}: train through surrogate, evaluate on analytic channel\n"
        f"Development seed {data['seed']}; {trials:,} symbol trials per point", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .14, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(args.out_dir / f"communication_curves.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    provenance = {name: hashlib.sha256((args.suite_dir/name).read_bytes()).hexdigest()
                  for name in ("results.json", "manifest.json")}
    if args.codebook_check is not None:
        provenance["codebook_check"] = hashlib.sha256(args.codebook_check.read_bytes()).hexdigest()
    if args.support_check is not None:
        provenance["support_check"] = hashlib.sha256(args.support_check.read_bytes()).hexdigest()
    (args.out_dir / "source_hashes.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    main()
