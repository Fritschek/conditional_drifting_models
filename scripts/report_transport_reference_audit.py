"""Recompute transport audit summaries from saved plans and fixed checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def report(folder, out):
    data = json.loads((folder/"projection_records.json").read_text())
    policies = json.loads((folder/"sspa_policy_records.json").read_text())
    gradients = json.loads((folder/"gradient_checks.json").read_text())
    equilibria = json.loads((folder/"equilibrium_checks.json").read_text())
    rows = data["rows"]
    if data["manifest"]["status"] != "complete" or len(rows) != 72 or len(policies["rows"]) != 12:
        raise ValueError("Incomplete reference audit")
    case_names = {r["case"] for r in rows}
    if len(case_names) != 8 or {(r["case"], r["epsilon"], r["iterations"]) for r in rows} != {
        (c, e, i) for c in case_names for e in (.001, .05, 1.) for i in (10, 30, 100)}:
        raise ValueError("Missing or duplicate projection cases")
    if {(r["policy"], r["iterations"]) for r in policies["rows"]} != {
        (p, i) for p in ("fixed_common", "shared_adaptive", "legacy_separate_adaptive", "historical_global_scale")
        for i in (10, 30, 100)}:
        raise ValueError("Missing or duplicate SSPA policies")
    references = {(p["case"], p["epsilon"]): p for p in data["plans"] if "reference" in p}
    production = {(p["case"], p["epsilon"], p["iterations"]): p for p in data["plans"] if "production" in p}
    residual = 0.
    def compare(actual, expected):
        nonlocal residual
        residual = max(residual, float(np.max(np.abs(np.asarray(actual)-expected))))
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    for row in rows:
        p = production[row["case"], row["epsilon"], row["iterations"]]["production"]
        plan, weights = np.array(p["coupling"]), np.array(p["row_weights"])
        n, m = plan.shape
        compare(row["raw_row_relative"], np.max(np.abs(n*plan.sum(1)-1)))
        compare(row["raw_column_relative"], np.max(np.abs(m*plan.sum(0)-1)))
        compare(row["normalized_column_relative"], np.max(np.abs(m*(weights/n).sum(0)-1)))
        ref = references[row["case"], row["epsilon"]]
        if ref["reference"]["converged"]:
            target = np.array(ref["target"])
            optimal = np.array(ref["reference"]["coupling"])
            compare(ref["reference"]["barycenter"], n*optimal@target)
            error = np.sqrt(np.mean((np.array(production[row["case"], row["epsilon"], row["iterations"]]["center"])-n*optimal@target)**2))
            compare(row["barycenter_rms_error"], error)
            if max(np.max(np.abs(n*optimal.sum(1)-1)), np.max(np.abs(m*optimal.sum(0)-1))) > 1.0001e-8:
                raise ValueError("Converged reference violates marginal tolerance")
        elif row["barycenter_rms_error"] is not None or row["coupling_l1_error"] is not None:
            raise ValueError("Unconverged reference used as ground truth")
    failures = sorted((c, e) for (c, e), p in references.items() if not p["reference"]["converged"])
    passed = gradients["passed"] and all(r["passed"] for r in equilibria)
    if not passed:
        raise ValueError("Mathematical controls failed")
    out.mkdir(parents=True, exist_ok=False)
    summary = {"reference_cases": len(references), "production_records": len(rows),
               "sspa_policy_records": len(policies["rows"]), "reference_failures": failures,
               "max_saved_plan_recomputation_difference": residual,
               "gradient_max_fd_error": gradients["finite_difference_max_absolute_error"],
               "gradient_mse_scaling_error": gradients["mse_identity_max_absolute_error"],
               "max_sspa_drift_error_at_10": max(r["drift_rms_error"] for r in policies["rows"] if r["iterations"] == 10),
               "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"checks.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# Small transport-reference audit", "",
             "Uniform masses, cost ||u-v||^2/2, KL relative to product masses.",
             "Float64 log-domain reference, both relative marginal errors <= 1e-8, cap 50,000 iterations.",
             "Production is float32 with its existing floor and denominator stabilizers. Training defaults are unchanged.", "",
             f"Reference cases: {len(references)}; unconverged cases: {failures}.",
             "Unconverged cases have no reported error against the intended OT solution.", "",
             "## Projection errors", "",
             "Barycenter errors are RMS per output coordinate, not normalized by cloud scale.",
             "Floor-reference error includes quantization of the production float32 kernel.", "",
             "| Case | Epsilon | Iterations | Raw row residual | Normalized target residual | Floor fraction | Barycenter error | Converged floored-kernel error |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        values = [r[k] for k in ("raw_row_relative", "normalized_column_relative", "floor_fraction",
                                 "barycenter_rms_error", "floor_only_barycenter_rms_error")]
        lines.append(f"| {r['case']} | {r['epsilon']} | {r['iterations']} | "
                     + " | ".join("unresolved" if v is None else f"{v:.6g}" for v in values)+" |")
    lines += ["", "## SSPA-shaped clouds", "",
              "Synthetic outputs at eight anchors, four outputs per cloud. No trained-generator trajectory.",
              "Policy differences below change the field; they are not themselves numerical errors.", "",
              "| Policy | Iterations | Cross epsilon | Self epsilon | Drift RMS error | Raw row residual | Reference field difference from fixed common |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in policies["rows"]:
        lines.append(f"| {r['policy']} | {r['iterations']} | " + " | ".join(f"{r[k]:.6g}" for k in
            ("cross_epsilon", "self_epsilon", "drift_rms_error", "max_raw_row_relative", "reference_difference_from_fixed_common_rms"))+" |")
    lines += ["", "## Mathematical checks", "",
              f"Parameter finite-difference error: {summary['gradient_max_fd_error']:.6g}.",
              f"Detached-MSE scaling identity error: {summary['gradient_mse_scaling_error']:.6g}.",
              f"Independent same-generator reference changes the sampled field by norm {gradients['independent_reference_drift_difference']:.6g}.",
              "The independent-reference field is not compared as an approximation of the same-batch objective gradient.",
              "Collapsed symmetric examples all have zero velocity and positive divergence; identical-law checks give zero divergence.", "",
              "These controls validate a restricted instantaneous gradient identity and expose numerical failure regimes.",
              "They neither explain the historical SSPA training degradation nor establish convergence of neural training.",
              "All full inputs, couplings and source snapshots are retained in the raw result directory."]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
