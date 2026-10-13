"""Independent portable-record audit; stdlib only, no training or weight loading."""

import ast
import cmath
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "Journal_version/evidence/application_local_20261013"
OUT = ROOT / "Journal_version/evidence/application_local_independent_audit_20261013.json"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-13), (a, b)


def main():
    unique_sources, summaries = {}, {}
    points = pairs = exports = 0
    for name in ("sspa", "ofdm_clean", "ofdm_interference"):
        folder = EVIDENCE / name
        m, r, a = [read(folder / f"{f}.json") for f in ("manifest", "results", "audit")]
        for filename, sha in read(folder / "checksums.json").items():
            assert digest(folder / filename) == sha, filename
            exports += 1
        assert digest(folder / "manifest.json") == r["manifest_sha256"] == a["manifest_sha256"]
        assert digest(folder / "results.json") == a["results_sha256"]
        assert digest(ROOT / "scripts/report_application_local.py") == a["reporter_sha256"]
        for filename, sha in m["sources"]["files"].items():
            assert digest(ROOT / filename) == sha, filename
            assert filename not in unique_sources or unique_sources[filename] == sha
            unique_sources[filename] = sha
        s = m["settings"]
        methods = {f"{f}_{n}" for f, grid in s["grids"].items() for n in grid}
        assert len(r["generators"]) == len(s["grids"])
        assert {g["family"] for g in r["generators"]} == set(s["grids"])
        for g in r["generators"]:
            c = g["cost"]
            outputs = c["counts"]["oracle_outputs"]
            if c["teacher"]:
                outputs += c["teacher"]["counts"]["oracle_outputs"]
            assert outputs == c["oracle_outputs_including_teacher"] == s["updates"] * s["batch"]
            assert abs(c["parameters"] / s["parameters"] - 1) < .1
            assert c["checkpoint_sha256"] == r["checkpoint_hashes"][g["family"] + "/generator.pt"]
        begin = points
        if name == "sspa":
            assert set(r["evaluations"]) == methods
            assert [b["ae_seed"] for b in r["blocks"]] == s["ae_seeds"]
            for block in r["blocks"]:
                assert set(block["models"]) == methods | {"analytic"}
                assert len(block["curves"]) == 3
                for curve in block["curves"]:
                    close(curve["input_ebno_db"] - curve["legacy_nominal_ebno_db"], 10 * math.log10(2))
                    close(curve["real_noise_variance"], (8 / 6) / (2 * 10 ** (curve["input_ebno_db"] / 10)))
                    assert set(curve["variants"]) == methods | {"analytic"}
                    for row in curve["variants"].values():
                        n = row["trials"]
                        assert n == 64 * s["eval_samples"]
                        assert sum(row["errors_per_message"]) == row["errors"]
                        assert sum(row["bit_errors_per_message"]) == row["bit_errors"]
                        close(row["ser"], row["errors"] / n)
                        close(row["ber"], row["bit_errors"] / (6 * n))
                        points += 1
                    seen = set()
                    for pair in curve["pairs"]:
                        key = frozenset((pair["a"], pair["b"]))
                        assert len(key) == 2 and key not in seen
                        seen.add(key)
                        assert sum(pair[k] for k in ("a_only_wrong", "b_only_wrong", "both_wrong", "neither_wrong")) == n
                        assert pair["a_only_wrong"] + pair["both_wrong"] == curve["variants"][pair["a"]]["errors"]
                        assert pair["b_only_wrong"] + pair["both_wrong"] == curve["variants"][pair["b"]]["errors"]
                        pairs += 1
                    assert len(seen) == len(curve["variants"]) * (len(curve["variants"]) - 1) // 2
        else:
            assert {v["method"] for v in r["rows"]} == methods | {"analytic"}
            assert len(r["rows"]) == len(methods) + 1
            records = r["rows"] + [dict(method="pilot_LMMSE", curves=r["conventional"])]
            activity_counts, nominal = {}, {}
            for row in records:
                assert [c["esno_db"] for c in row["curves"]] == s["esnos"]
                for curve in row["curves"]:
                    groups = curve["groups"]
                    assert groups["all"]["blocks"] == s["eval_blocks"]
                    for key in ("blocks", "block_errors", "symbol_errors", "bit_errors", "squared_block_bit_errors"):
                        assert groups["all"][key] == groups["active"][key] + groups["inactive"][key]
                    for v in groups.values():
                        b = v["blocks"]
                        assert v["symbol_trials"] == b * 56 and v["bit_trials"] == b * 112
                        for key, bound in (("block_errors", b), ("symbol_errors", 56*b), ("bit_errors", 112*b)):
                            assert isinstance(v[key], int) and 0 <= v[key] <= bound
                        if b:
                            close(v["ber"], v["bit_errors"] / (112*b))
                            close(v["ser"], v["symbol_errors"] / (56*b))
                            close(v["bler"], v["block_errors"] / b)
                            if b > 1:
                                se = math.sqrt(max(0, v["squared_block_bit_errors"] - v["bit_errors"]**2 / b) / (b*(b-1))) / 112
                                close(v["ber_mc_se_blocks"], se)
                            if not v["block_errors"]:
                                close(v["zero_block_error_upper95"], -math.expm1(math.log(.05) / b))
                        else:
                            assert all(v[k] is None for k in ("ber", "ser", "bler", "ber_mc_se_blocks"))
                    activity_counts.setdefault(curve["esno_db"], groups["active"]["blocks"])
                    assert activity_counts[curve["esno_db"]] == groups["active"]["blocks"]
                    points += 1
                    if curve["esno_db"] == 10:
                        nominal[row["method"]] = dict(ber=groups["all"]["ber"],
                            mc_se=groups["all"]["ber_mc_se_blocks"], inactive=groups["inactive"]["ber"],
                            active=groups["active"]["ber"])
                if "metrics" in row:
                    d = row["metrics"]
                    nominal[row["method"]].update(swd_over_floor=d["swd"]/d["swd_floor"],
                        crosscov_over_floor=d["pilot_data_covariance_error"]/d["pilot_data_covariance_floor"],
                        learned_energy_exceedance=d["learned_exceedance"], true_energy_exceedance=d["analytic_exceedance"])
            if name == "ofdm_interference":
                frac = activity_counts[10] / s["eval_blocks"]
                ref = nominal["analytic"]
                for k, row in nominal.items():
                    if k == "analytic":
                        continue
                    inactive_gap = (1-frac) * (row["inactive"]-ref["inactive"])
                    active_gap = frac * (row["active"]-ref["active"])
                    close(inactive_gap + active_gap, row["ber"]-ref["ber"])
                    row.update(inactive_contribution_to_gap=inactive_gap, active_contribution_to_gap=active_gap,
                               inactive_fraction_of_gap=inactive_gap/(row["ber"]-ref["ber"]))
            summaries[name] = nominal
        assert points - begin == a["curve_points"]
    folder = EVIDENCE / "sspa"
    inspection = read(folder / "codebook_inspection.json")
    assert inspection["parent_results_sha256"] == digest(folder / "results.json")
    assert inspection["parent_manifest_sha256"] == digest(folder / "manifest.json")
    assert inspection["script_sha256"] == digest(ROOT / "scripts/inspect_sspa_application_codebooks.py")
    assert digest(folder / "codebook_inspection.json") == "3b591b0268f0c3c67f70c2b8321f3aaad5fc6a8ccd422c4ef7b2aeb4a9cfa3eb"
    for b in inspection["blocks"]:
        for r in b["rows"]:
            assert r["trials"] == inspection["samples_per_message"] * 64
            assert sum(r["per_message"]["neural"]) == r["neural_errors"] == r["neural_only_wrong"] + r["both_wrong"]
            assert sum(r["per_message"]["nearest"]) == r["nearest_mean_errors"] == r["nearest_only_wrong"] + r["both_wrong"]
            close(r["neural_ser"], r["neural_errors"]/r["trials"])
            close(r["nearest_mean_ser"], r["nearest_mean_errors"]/r["trials"])
    constants = {}
    for node in ast.parse((ROOT / "conditional_drifting/application_channels.py").read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("TDLC_DELAYS", "TDLC_POWERS_DB"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    delays = [v * 300e-9 for v in constants["TDLC_DELAYS"]]
    powers = [10**(v/10) for v in constants["TDLC_POWERS_DB"]]
    powers = [v/sum(powers) for v in powers]
    pilot_cross = min(max(abs(sum(p*cmath.exp(-2j*math.pi*(k-j)*30000*t)
                                for p, t in zip(powers, delays))) for j in range(0,64,8))
                      for k in range(64) if k % 8)
    assert pilot_cross > .94
    widths = [min(range(8,2049), key=lambda h: abs(h*h+h*(128+8384+2)+8384-target)) for target in (600000,300000)]
    result = dict(scope="Portable records and current source hashes only; checkpoints and CUDA not rerun",
        portable_export_hashes_verified=exports, posthoc_export_hashes_verified=1,
        unique_local_source_hashes_verified=len(unique_sources), physical_curve_points=points,
        sspa_paired_discordance_records=pairs, sources=unique_sources, ofdm_nominal=summaries,
        gaussian_hidden_width=widths[0], mixture_component_hidden_width=widths[1],
        free_real_data_input_dimension=112, minimum_max_pilot_crosscov_magnitude=pilot_cross,
        missing_preliminary_receiver_export=not (EVIDENCE / "receivers").exists(),
        audit_script_sha256=digest(Path(__file__)))
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("sources", "ofdm_nominal")}, indent=2))


if __name__ == "__main__":
    main()
