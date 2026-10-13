"""Audit and summarize the local common-coverage and structured OFDM pilots."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.baselines.official_sources import sha256
from conditional_drifting.baselines.fast_evaluation import write_json


def read(path):
    return json.loads(path.read_text())


def check_counts(row, symbols, bits):
    for key, total in (("errors", symbols), ("bit_errors", bits)):
        if not isinstance(row[key], int) or not 0 <= row[key] <= total:
            raise ValueError(f"Invalid {key}")
    if row["trials"] != symbols or abs(row["ser"]-row["errors"]/symbols) > 1e-14 or abs(row["ber"]-row["bit_errors"]/bits) > 1e-14:
        raise ValueError("Error rate/count mismatch")


def audit(suite, kind):
    m, r = read(suite/"manifest.json"), read(suite/"results.json")
    if r["manifest_sha256"] != sha256(suite/"manifest.json"):
        raise ValueError("Manifest mismatch")
    for name, digest in m["sources"]["files"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Numerical source changed: {name}")
    for name, digest in r["checkpoint_hashes"].items():
        if sha256(suite/name) != digest:
            raise ValueError(f"Checkpoint changed: {name}")
    s = m["settings"]
    expected = {f"{family}_{steps}" for family, grid in s["grids"].items() for steps in grid}
    families = [g["family"] for g in r["generators"]]
    if set(families) != set(s["grids"]) or len(families) != len(s["grids"]):
        raise ValueError("Missing or duplicate generator")
    for g in r["generators"]:
        if g["cost"]["oracle_outputs_including_teacher"] != s["updates"]*s["batch"]:
            raise ValueError("Oracle query budget mismatch")
        if abs(g["cost"]["parameters"]/s["parameters"]-1) > .1:
            raise ValueError("Capacity exceeds declared tolerance")
    lines = [f"# Local {kind} development", "", f"Profile: {s['profile']}; seed: {s['seed']}.", "",
             "Development outcomes only. Final confirmation and model selection remain separate.", "",
             "## Generator cost", "", "| Family | Parameters | Oracle outputs (including teacher) | Training seconds | Mean input energy per anchor |",
             "|---|---:|---:|---:|---:|"]
    for g in r["generators"]:
        c = g["cost"]
        lines.append(f"| {g['family']} | {c['parameters']} | {c['oracle_outputs_including_teacher']} | {c['train_seconds']:.2f} | {c['mean_query_energy']:.4g} |")
    count = 0
    if kind == "sspa":
        if set(r["evaluations"]) != expected or len(r["blocks"]) != len(s["ae_seeds"]):
            raise ValueError("Incomplete SSPA panel")
        lines += ["", "Physical input-referenced Eb/N0 is shown. These retain the previous physical",
                  "variance, with labels 3.0103 dB above the historical software parameter.", "",
                  "| Method | Physical Eb/N0 (dB) | SER | BER | AE training seconds |",
                  "|---|---:|---:|---:|---:|"]
        for b in r["blocks"]:
            if set(b["models"]) != expected | {"analytic"} or len(b["curves"]) != 3:
                raise ValueError("Incomplete AE/curve grid")
            for curve in b["curves"]:
                if set(curve["variants"]) != expected | {"analytic"}:
                    raise ValueError("Missing curve variant")
                for name, row in curve["variants"].items():
                    trials = 64*s["eval_samples"]
                    check_counts(row, trials, 6*trials)
                    if sum(row["errors_per_message"]) != row["errors"] or sum(row["bit_errors_per_message"]) != row["bit_errors"]:
                        raise ValueError("Per-message totals differ")
                    count += 1
                    lines.append(f"| {name} | {curve['input_ebno_db']:.4f} | {row['ser']:.6g} | {row['ber']:.6g} | {b['models'][name]['train_seconds']:.2f} |")
                pairs = curve["pairs"]
                if len(pairs) != len(curve["variants"])*(len(curve["variants"])-1)//2:
                    raise ValueError("Incomplete paired comparisons")
                for pair in pairs:
                    if sum(pair[k] for k in ("a_only_wrong", "b_only_wrong", "both_wrong", "neither_wrong")) != trials:
                        raise ValueError("Discordance totals differ")
                    if pair["a_only_wrong"]+pair["both_wrong"] != curve["variants"][pair["a"]]["errors"]:
                        raise ValueError("Paired a count differs")
                    if pair["b_only_wrong"]+pair["both_wrong"] != curve["variants"][pair["b"]]["errors"]:
                        raise ValueError("Paired b count differs")
        lines += ["", "| Method | Ordinary-input conditional SWD/floor |", "|---|---:|"]
        for name, evaluation in r["evaluations"].items():
            d = evaluation["distribution"]
            lines.append(f"| {name} | {d['conditional']['swd']/d['analytic_floor']['swd']:.4g} |")
    else:
        if {row["method"] for row in r["rows"]} != expected | {"analytic"} or len(r["rows"]) != len(expected)+1:
            raise ValueError("Incomplete OFDM panel")
        lines += ["", f"Learned constellation: {s['learned_constellation']}. All receiver tests use the physical channel.", "",
                  "| Method | Es/N0 | BER | Block error rate | Inactive BER | Active BER | Receiver seconds |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        all_rows = r["rows"]+[dict(method="pilot_LMMSE_fixed_QPSK", curves=r["conventional"], receiver=dict(seconds=0))]
        for row in all_rows:
            if [c["esno_db"] for c in row["curves"]] != s["esnos"]:
                raise ValueError("Incomplete physical SNR curve")
            for curve in row["curves"]:
                groups = curve["groups"]
                if groups["all"]["blocks"] != s["eval_blocks"]:
                    raise ValueError("Wrong block budget")
                for key in ("blocks", "block_errors", "symbol_errors", "bit_errors"):
                    if groups["all"][key] != groups["active"][key]+groups["inactive"][key]:
                        raise ValueError("Activity count mismatch")
                for group in groups.values():
                    b = group["blocks"]
                    if group["bit_trials"] != b*112 or group["symbol_trials"] != b*56:
                        raise ValueError("Wrong OFDM trial count")
                    if b and (abs(group["ber"]-group["bit_errors"]/(112*b)) > 1e-14 or
                              abs(group["bler"]-group["block_errors"]/b) > 1e-14):
                        raise ValueError("Rate/count mismatch")
                count += 1
                active = "n/a" if groups["active"]["ber"] is None else f"{groups['active']['ber']:.6g}"
                lines.append(f"| {row['method']} | {curve['esno_db']:g} | {groups['all']['ber']:.6g} | {groups['all']['bler']:.6g} | {groups['inactive']['ber']:.6g} | {active} | {row['receiver']['seconds']:.2f} |")
        lines += ["", "| Method | Conditional SWD/floor | Pilot/data covariance error/floor |", "|---|---:|---:|"]
        for row in r["rows"]:
            d = row["metrics"]
            lines.append(f"| {row['method']} | {d['swd']/d['swd_floor']:.4g} | {d['pilot_data_covariance_error']/d['pilot_data_covariance_floor']:.4g} |")
        lines += ["", "The conventional receiver knows second-order channel statistics but not the hidden",
                  "realization or activity. Its rows use fixed QPSK even in learned-constellation runs.",
                  "BER uncertainty is calculated across independent blocks, not independent bits."]
    return "\n".join(lines)+"\n", dict(kind=kind, curve_points=count, complete=True,
        manifest_sha256=r["manifest_sha256"], results_sha256=sha256(suite/"results.json"),
        sources_verified=len(m["sources"]["files"]), checkpoints_verified=len(r["checkpoint_hashes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("sspa", "ofdm"), required=True)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args()
    report, result = audit(args.suite_dir, args.kind)
    result["reporter_sha256"] = sha256(__file__)
    (args.suite_dir/"report.md").write_text(report)
    write_json(args.suite_dir/"audit.json", result)
    if args.export_dir:
        files = ("manifest.json", "results.json", "report.md", "audit.json")
        for name in files:
            target = args.export_dir/name
            if target.exists() and sha256(target) != sha256(args.suite_dir/name):
                raise FileExistsError(f"Refusing to replace evidence: {target}")
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copyfile(args.suite_dir/name, args.export_dir/name)
        write_json(args.export_dir/"checksums.json", {name: sha256(args.export_dir/name) for name in files})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
