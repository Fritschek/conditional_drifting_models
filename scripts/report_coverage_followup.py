"""Compact audited tables for the completed coverage/sampler follow-up."""

import argparse
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.baselines.fast_evaluation import write_json
from conditional_drifting.baselines.official_sources import sha256
from scripts.coverage_followup_common import read
from scripts.report_sspa_coverage_replication import summarize


def coverage(suite):
    summary, flat = summarize(suite)
    result = read(suite / "results.json")
    lines = ["# Coverage replication", "", f"Profile: {summary['profile']}. "
             f"{summary['generators']} generators, {summary['ae_runs']} AEs.", "",
        "Primary: broad-minus-normal physical SER at 8 dB, averaged within generator seed.",
        "Negative is improvement. Generator seeds, not individual AEs, are the replication units.", "",
        "| Family | Generator seed | Normal mean SER | Broad mean SER | Difference |",
        "|---|---:|---:|---:|---:|"]
    for family in ("gaussian", "sinkhorn"):
        for group in next(p for p in summary["primary"] if p["family"] == family)["generator_blocks"]:
            effects = [e for e in summary["paired_effects"] if e["family"] == family and e["generator_seed"] == group["generator_seed"]]
            normal = statistics.mean(e["normal_ser"] for e in effects)
            broad = statistics.mean(e["broad_ser"] for e in effects)
            lines.append(f"| {family} | {group['generator_seed']} | {normal:.6g} | {broad:.6g} | {broad-normal:+.6g} |")
    lines += ["", "## All nominal-SNR pairs", "",
        "| Family | Generator seed | AE seed | Normal SER | Broad SER | Difference |",
        "|---|---:|---:|---:|---:|---:|"]
    for e in summary["paired_effects"]:
        lines.append(f"| {e['family']} | {e['generator_seed']} | {e['ae_seed']} | {e['normal_ser']:.6g} | {e['broad_ser']:.6g} | {e['broad_minus_normal']:+.6g} |")
    lines += ["", "## Conditional fidelity and cost", "",
        "| Seed | Variant | Ordinary SWD/floor | Radius-10 mean RMSE | Max-radius variance/truth | Mean query energy | Train seconds |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for g in result["generators"]:
        ev, cost = g["evaluation"], g["cost"]
        ordinary = ev["random_inputs"]
        ratio = ordinary["conditional"]["swd"]/ordinary["analytic_floor"]["swd"]
        r10 = next(r for r in ev["radial"] if r["radius"] == 10.)
        lines.append(f"| {g['generator_seed']} | {g['variant']} | {ratio:.4g} | {r10['mean_rmse']:.4g} | {ev['radial'][-1]['variance_ratio']:.4g} | {cost['observed_mean_squared_input_norm']:.4g} | {cost['train_seconds']:.2f} |")
    lines += ["", "## Transfer optimism and codewords", "",
        "| Seed | AE seed | Variant | Physical SER | Surrogate SER | Maximum codeword norm | AE seconds |",
        "|---:|---:|---|---:|---:|---:|---:|"]
    for b in result["blocks"]:
        nominal = b["curves"][-1]["variants"]
        for variant, row in b["rows"].items():
            lines.append(f"| {b['generator_seed']} | {b['ae_seed']} | {variant} | {nominal[variant]['ser']:.6g} | {row['surrogate_nominal']['ser']:.6g} | {max(row['codebook_norms']):.4g} | {row['ae_train_seconds']:.2f} |")
    lines += ["", "Surrogate/physical evaluations here use independent noise. Paired physical comparisons",
              "use common noise and retain discordance counts in results.json. Expected query energy",
              "is 6.0833x higher under the broader law; no claim of equal RF measurement energy.", ""]
    return "\n".join(lines), summary


def samplers(suite):
    result = read(suite / "results.json")
    manifest = read(suite / "manifest.json")
    if result["manifest_sha256"] != sha256(suite / "manifest.json"):
        raise ValueError("Sampler manifest mismatch")
    expected = {(f, n) for f, grid in manifest["grids"].items() for n in grid}
    if {(r['family'], r['nfe']) for r in result['rows']} != expected or len(result['rows']) != len(expected):
        raise ValueError("Incomplete sampler panel")
    for name, digest in manifest["sources"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Sampler source changed: {name}")
    lines = ["# Saved-checkpoint sampler panel", "", f"Profile: {manifest['profile']}. No retraining or NFE selection.", "",
        "| Family | NFE | Conditional SWD/floor | Variance/truth | Sampling ms/batch | Forward+input backward ms/batch | Encoder cosine | Reference resolved |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["rows"]:
        path = suite / f"{r['family']}_{r['nfe']}.pt"
        if sha256(path) != r["panel_sha256"]:
            raise ValueError("Sampler tensor panel changed")
        d, g = r["distribution"], r["early_codec_gradients"]["encoder"]
        cosine = "undefined" if g["alignment"]["cosine"] is None else f"{g['alignment']['cosine']:.4g}"
        times = {t["operation"]: t["mean_ms"] for t in r["timing"] if t["batch"] == manifest["settings"]["ae_batch"]}
        lines.append(f"| {r['family']} | {r['nfe']} | {d['conditional']['swd']/d['analytic_floor']['swd']:.4g} | {r['moments']['variance_ratio']:.4g} | {times['sampling']:.4g} | {times['forward_and_input_backward']:.4g} | {cosine} | {g['reference_resolved']} |")
    lines += ["", f"Timing batch: {manifest['settings']['ae_batch']}. These are batch times, not per-sample latency.",
              "Batch-one measurements and full raw timing repeats are retained in results.json.", "",
              "| Family | NFE | Output RMSE vs finest | Pathwise Jacobian RMSE vs finest |", "|---|---:|---:|---:|"]
    for r in result["rows"]:
        d = r["relative_to_finest"]
        lines.append(f"| {r['family']} | {r['nfe']} | {d['paired_output_rmse']:.6g} | {d['paired_jacobian_rmse']:.6g} |")
    lines += ["", "The finest sampler is not an analytic reference. Differences test finite-grid",
              "self-consistency only. Resolved flags qualify MC reference precision, not model correctness.", ""]
    return "\n".join(lines), dict(profile=manifest["profile"], rows=len(result["rows"]), complete=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--kind", choices=("coverage", "samplers"), required=True)
    args = p.parse_args()
    report, summary = (coverage if args.kind == "coverage" else samplers)(args.suite_dir)
    summary.update(source_results_sha256=sha256(args.suite_dir / "results.json"), report_script_sha256=sha256(__file__))
    write_json(args.suite_dir / "report_audit.json", summary)
    (args.suite_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
