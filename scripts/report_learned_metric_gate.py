"""Report all same-input contrasts without pooling anchors into model replicates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_learned_metric_gate import CHEAP, METHODS, contrasts, refinement_decisions, summary


def validate(data):
    manifest, rows, losses = data["manifest"], data["rows"], data["losses"]
    if manifest.get("status") != "complete" or manifest["config"]["preflight_only"]:
        raise ValueError("Require a completed main panel, not preflight output")
    expected = {(512, a, m, r) for a in range(3) for m in METHODS for r in range(8)}
    refined = [d["anchor"] for d in manifest["refinement"] if d["refine"]]
    expected |= {(2048, a, m, r) for a in refined for m in METHODS for r in range(8)}
    key = lambda r: (r["samples"], r["anchor"], r["method"], r["repeat"])
    if len(rows) != len(expected) or {key(r) for r in rows} != expected:
        raise ValueError("Incomplete or duplicated metric panel")
    probes = {p["name"] for p in manifest["probes"]}
    expected_losses = {k+(p,) for k in expected for p in probes}
    if len(losses) != len(expected_losses) or {key(r)+(r["probe"],) for r in losses} != expected_losses:
        raise ValueError("Incomplete or duplicated task panel")
    if refinement_decisions(contrasts(rows, losses, 512)) != manifest["refinement"]:
        raise ValueError("Refinement decisions do not reproduce")
    return {"metric_records": len(rows), "task_records": len(losses),
            "learned_metric_records": sum(r["method"] != "analytic" for r in rows),
            "refinement_reproduced": True}


def csv_write(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(data):
    rows, losses = data["rows"], data["losses"]
    metrics, targets = [], []
    keys = sorted({(r["samples"], r["anchor"], r["method"]) for r in rows})
    scalar_names = [k for k, v in rows[0].items() if isinstance(v, (int, float))
                    and k not in {"anchor", "repeat", "samples"}]
    for n, anchor, method in keys:
        group = [r for r in rows if (r["samples"], r["anchor"], r["method"]) == (n, anchor, method)]
        base = {"samples": n, "anchor": anchor, "method": method}
        row = dict(base)
        for name in scalar_names:
            row.update({name+"_"+k: v for k, v in summary([r[name] for r in group]).items()})
        metrics.append(row)
        for probe in data["manifest"]["probes"]:
            task = [r for r in losses if (r["samples"], r["anchor"], r["method"], r["probe"])
                    == (n, anchor, method, probe["name"])]
            gs = np.array([(np.array(r["gradient_split1"])+np.array(r["gradient_split2"]))/2 for r in task])
            exact = np.array(task[0]["reference_gradient"])
            result = base | {"probe": probe["name"], "kind": probe["kind"],
                             "pooled_absolute_error": float(np.linalg.norm(gs.mean(0)-exact)),
                             "gradient_mean_mc_se_l2": float(np.linalg.norm(gs.std(0, ddof=1)/np.sqrt(len(gs)))),
                             "reference_gradient_norm": float(np.linalg.norm(exact))}
            for name in ("absolute_gradient_error", "cross_squared_error", "self_squared_error", "estimated_noise_squared"):
                result.update({name+"_"+k: v for k, v in summary([r[name] for r in task]).items()})
            targets.append(result)
    return metrics, targets


def report(data, out):
    checks = validate(data)
    out.mkdir(parents=True, exist_ok=False)
    metrics, targets = summarize(data)
    csv_write(out/"metric_summary.csv", metrics)
    csv_write(out/"task_summary.csv", targets)
    all_contrasts = [r for n in sorted({r["samples"] for r in data["rows"]})
                     for r in contrasts(data["rows"], data["losses"], n)]
    final_n = {a: max(r["samples"] for r in data["rows"] if r["anchor"] == a) for a in range(3)}
    final = [r for r in all_contrasts if r["samples"] == final_n[r["anchor"]]]
    counts = {"total_contrasts": len(final),
              "resolved_target_contrasts": sum(bool(r["target"]["ordering"]) for r in final),
              "kernel_agrees": sum(r["kernel_agrees"] for r in final),
              "candidate_added_information": sum(r["candidate_added_information"] for r in final)}
    comparisons = {}
    for name in CHEAP+("rbf_cross_trace",):
        valid = [r for r in final if r["target"]["ordering"]]
        comparisons[name] = {
            "agrees": sum(r["scores"][name]["ordering"] == r["target"]["ordering"] for r in valid),
            "opposes": sum(r["scores"][name]["ordering"] == -r["target"]["ordering"] for r in valid),
            "unresolved": sum(r["scores"][name]["ordering"] == 0 for r in valid)}
    checks.update(counts)
    checks["comparator_counts"] = comparisons
    checks["final_samples_by_anchor"] = final_n
    checks["analytic_nulls_outside_descriptive_3se"] = {
        "rbf_derivative": [{"anchor": r["anchor"], "samples": r["samples"]}
                           for r in metrics if r["method"] == "analytic"
                           and abs(r["rbf_cross_trace_mean"]) > 3*r["rbf_cross_trace_mc_se"]+1e-12],
        "task_squared_error": [{"anchor": r["anchor"], "samples": r["samples"], "probe": r["probe"]}
                               for r in targets if r["method"] == "analytic"
                               and abs(r["cross_squared_error_mean"]) > 3*r["cross_squared_error_mc_se"]+1e-12]}
    checks["report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/"checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    (out/"all_contrasts.json").write_text(json.dumps(all_contrasts, indent=2)+"\n")
    (out/"final_contrasts.json").write_text(json.dumps(final, indent=2)+"\n")
    lines = ["# Same-input SSPA metric development test", "",
             "Seed 7, three eight-dimensional inputs, eight fixed loss probes. No decoder training.",
             "Reference loss gradients use Gaussian integration through the SSPA mean Jacobian.",
             "Score and task samples are independent. Cross scores retain negative values.", "",
             f"Coverage: {checks['metric_records']} metric records ({checks['learned_metric_records']} learned), "
             f"{checks['task_records']} task records. Final sample counts: {final_n}.",
             "Uncertainty below is MC SE across eight repetitions, not across independently trained models.",
             "The 3-SE resolution rule is descriptive and uncalibrated; counts reuse model pairs across probes.", "",
             "## Predeclared contrasts", "",
             f"Resolved target orderings: {counts['resolved_target_contrasts']} / {len(final)}.",
             f"Candidate kernel added-information contrasts: {counts['candidate_added_information']} / {len(final)}.", "",
             "| Comparator | Agrees with resolved target | Opposes | Unresolved |", "|---|---:|---:|---:|"]
    for name, values in comparisons.items():
        lines.append(f"| {name} | {values['agrees']} | {values['opposes']} | {values['unresolved']} |")
    lines += ["", "These counts describe this panel, not independent successes or a ranking benchmark.",
              "An added-information contrast needs new model seeds and new probes before any selection claim.",
              "A null count supplies no evidence of a kernel-specific advantage at this resolution.", "",
              "## Per-input scores", "",
              "Values are means +/- MC SE; self and cross columns are squared Hilbert-Schmidt quantities.",
              "The noise term is self minus signed cross. It is candidate-specific, not subtracted using an analytic floor.", "",
              "| Input | Model | SWD | RBF derivative cross | RBF derivative noise | Polynomial derivative cross |",
              "|---|---|---:|---:|---:|---:|"]
    def fmt(row, name):
        return f"{row[name+'_mean']:.5g} +/- {row[name+'_mc_se']:.2g}"
    for row in metrics:
        if row["samples"] == final_n[row["anchor"]]:
            lines.append(f"| {row['anchor']} | {row['method']} | " + " | ".join(fmt(row, k) for k in
                ("swd", "rbf_cross_trace", "rbf_estimated_noise_trace", "moments_cross_trace"))+" |")
    lines += ["", "## Absolute expected-gradient errors", "",
              "Norm of the pooled gradient error across repetitions, followed by the L2 norm of coordinate MC SEs.",
              "The latter describes sampling uncertainty, not a scalar norm confidence interval.", "",
              "| Input | Probe | Analytic control | Conditional Sinkhorn | WGAN | DDIM-10 |",
              "|---|---|---:|---:|---:|---:|"]
    for a in range(3):
        for p in data["manifest"]["probes"]:
            vals = []
            for method in METHODS:
                row = next(r for r in targets if (r["samples"], r["anchor"], r["method"], r["probe"])
                           == (final_n[a], a, method, p["name"]))
                vals.append(f"{row['pooled_absolute_error']:.5g} +/- {row['gradient_mean_mc_se_l2']:.2g}")
            lines.append(f"| {a} | {p['name']} | " + " | ".join(vals)+" |")
    lines += ["", "## Cost", "", "Sampled outputs include a full input Jacobian for each draw.",
              "Shared analytic score draws are counted once. Task and score draws are separate.", "",
              "```json", json.dumps({k: data["manifest"][k] for k in
                 ("sampled_outputs", "timing_seconds", "elapsed_seconds")}, indent=2), "```", "",
              "N=512 and N=2048 raw summaries are both retained in the CSVs. No levels were pooled.",
              "The analytic candidate supplies the independent task-noise control; no noisy reference task gradient is needed.",
              "Quadratic probes test the moment baseline. The higher-order probes do not constitute downstream SER/BER."]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(checks, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report(json.loads(args.results.read_text()), args.out_dir)


if __name__ == "__main__":
    main()
