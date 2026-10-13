"""Independent standard-library audit of the portable coverage/sampler records."""

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Journal_version/evidence/coverage_followup_20261013"
VARIANTS = ("analytic", "gaussian_normal", "gaussian_broad", "sinkhorn_normal", "sinkhorn_broad")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_point(p):
    n, k, b = (p[x] for x in ("trials", "errors", "bit_errors"))
    assert n == 1048576 and isinstance(k, int) and isinstance(b, int)
    assert 0 <= k <= n and k <= b <= 6*k
    assert p["ser"] == k/n and p["ber"] == b/(6*n)
    if "errors_per_message" in p:
        es, bs = p["errors_per_message"], p["bit_errors_per_message"]
        assert len(es) == len(bs) == 64 and sum(es) == k and sum(bs) == b
        assert all(0 <= e <= n//64 and e <= bits <= 6*e for e, bits in zip(es, bs))


def close(x, y):
    assert math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-18), (x, y)


def main():
    index = read(BASE/"checksums.json")
    for name, digest in index["files"].items():
        assert sha(BASE/name) == digest, name
    datasets, manifests, sources = {}, {}, {}
    for kind in ("coverage", "samplers"):
        folder = BASE/kind
        datasets[kind], manifests[kind] = read(folder/"results.json"), read(folder/"manifest.json")
        assert datasets[kind]["manifest_sha256"] == sha(folder/"manifest.json")
        report_audit = read(folder/"report_audit.json")
        assert report_audit["source_results_sha256"] == sha(folder/"results.json")
        assert report_audit["report_script_sha256"] == sha(ROOT/"scripts/report_coverage_followup.py")
        for name, digest in {**manifests[kind]["parent"]["adapter_files"], **manifests[kind]["sources"]}.items():
            assert sha(ROOT/name) == digest, name
            sources[name] = digest
    c, s = datasets["coverage"], datasets["samplers"]
    plan = manifests["coverage"]["plan"]
    assert plan["profile"] == "paper" and manifests["samplers"]["profile"] == "paper"
    seeds = plan["generator_seeds"]
    generators = {(g["generator_seed"], g["variant"]): g for g in c["generators"]}
    assert len(generators) == len(c["generators"]) == 12
    assert set(generators) == set(itertools.product(seeds, VARIANTS[1:]))
    for seed in seeds:
        for family in ("gaussian", "sinkhorn"):
            n, b = (generators[seed, family+"_"+law]["cost"] for law in ("normal", "broad"))
            assert n["config"] == b["config"] and n["counts"] == b["counts"]
            assert n["config"]["seed"] == seed and n["config"]["updates"] == 40000
            assert n["config"]["inference_weights"] == "ema" and n["counts"]["oracle_outputs"] == 40960000
            for cost in (n, b):
                close(cost["input_energy_sum"]/cost["counts"]["input_anchors"], cost["observed_mean_squared_input_norm"])
    blocks = {(b["generator_seed"], b["ae_seed"]): b for b in c["blocks"]}
    assert len(blocks) == len(c["blocks"]) == 9
    assert set(blocks) == {(seed, a) for seed in seeds for a in plan["ae_seeds"][str(seed)]}
    pairs, flat, primary_pairs = [], [], []
    for (seed, ae), block in blocks.items():
        assert set(block["rows"]) == set(VARIANTS)
        for name, row in block["rows"].items():
            assert row["generator_seed"] == seed and row["ae_seed"] == ae and row["variant"] == name
            codes = row["codebook"]
            assert len(codes) == 64 and all(len(code) == 8 for code in codes)
            energy = st.mean(sum(x*x for x in code) for code in codes)
            assert abs(energy-8) < 1e-5
            assert abs(st.mean(x for code in codes for x in code)) < 1e-6
            for code, norm in zip(codes, row["codebook_norms"]):
                assert math.isclose(sum(x*x for x in code), norm*norm, rel_tol=5e-7)
            check_point(row["surrogate_nominal"])
        assert [x["ebno_db"] for x in block["curves"]] == [4., 6., 8.]
        for curve in block["curves"]:
            assert set(curve["variants"]) == set(VARIANTS)
            for name, p in curve["variants"].items():
                check_point(p)
                flat.append(dict(generator_seed=seed, ae_seed=ae, variant=name, ebno_db=curve["ebno_db"],
                    ser=p["ser"], ber=p["ber"], errors=p["errors"], trials=p["trials"]))
            assert len(curve["pairs"]) == 10
            assert {(p["a"], p["b"]) for p in curve["pairs"]} == set(itertools.combinations(VARIANTS, 2))
            for pair in curve["pairs"]:
                a, b = (curve["variants"][pair[k]] for k in ("a", "b"))
                aw, bw, both, neither = (pair[k] for k in ("a_only_wrong", "b_only_wrong", "both_wrong", "neither_wrong"))
                assert all(isinstance(k, int) and k >= 0 for k in (aw, bw, both, neither))
                assert aw+both == a["errors"] and bw+both == b["errors"]
                n = a["trials"]
                assert aw+bw+both+neither == pair["trials"] == n
                delta, se = (aw-bw)/n, math.sqrt(((aw+bw)/n-((aw-bw)/n)**2)/(n-1))
                close(delta, pair["ser_a_minus_b"])
                close(se, pair["pooled_paired_mc_se"])
                pairs.append(pair)
                if curve["ebno_db"] == 8 and pair["b"] == pair["a"].replace("_normal", "_broad"):
                    primary_pairs.append(dict(generator_seed=seed, ae_seed=ae, family=pair["a"].split("_")[0],
                        normal_errors=a["errors"], broad_errors=b["errors"], broad_minus_normal=-delta,
                        pooled_paired_mc_se=se, descriptive_change_over_mc_se=-delta/se if se else None))
    exported = list(csv.DictReader((BASE/"coverage/curves.csv").open()))
    assert len(exported) == len(flat) == 135
    for x, y in zip(exported, flat):
        assert set(x) == set(y)
        for key in y:
            assert (x[key] == y[key]) if key == "variant" else (float(x[key]) == y[key])
    summary = read(BASE/"coverage/summary.json")
    assert summary["aggregation_script_sha256"] == sha(ROOT/"scripts/report_sspa_coverage_replication.py")
    primary, by_snr = {}, {}
    for family in ("gaussian", "sinkhorn"):
        primary[family] = {}
        for snr in (4., 6., 8.):
            groups = []
            for seed in seeds:
                points = [next(v for v in block["curves"] if v["ebno_db"] == snr)["variants"]
                          for (g, ae), block in blocks.items() if g == seed]
                n, b = (st.mean(p[family+"_"+law]["ser"] for p in points) for law in ("normal", "broad"))
                groups.append(dict(generator_seed=seed, normal=n, broad=b, difference=b-n, ratio=n/b))
            by_snr[f"{family}_{snr:g}"] = groups
            if snr == 8:
                primary[family] = dict(normal_mean=st.mean(g["normal"] for g in groups), broad_mean=st.mean(g["broad"] for g in groups),
                    mean_change=st.mean(g["difference"] for g in groups), between_generator_sd=st.stdev(g["difference"] for g in groups))
                expected = next(v for v in summary["primary"] if v["family"] == family)
                for own, saved in (("normal_mean", "normal_mean_ser"), ("broad_mean", "broad_mean_ser"),
                                   ("mean_change", "mean_change"), ("between_generator_sd", "between_generator_sd")):
                    close(primary[family][own], expected[saved])
                close(primary[family]["normal_mean"]/primary[family]["broad_mean"], expected["normal_divided_by_broad"])
                assert sum(p["broad_minus_normal"] < 0 for p in primary_pairs if p["family"] == family) == expected["numerically_improved_pairs"]
    sampler_summary = []
    assert {(r["family"], r["nfe"]) for r in s["rows"]} == {(f, n) for f, grid in manifests["samplers"]["grids"].items() for n in grid}
    assert len(s["rows"]) == 6
    for r in s["rows"]:
        for k in ("code", "encoder"):
            g = r["early_codec_gradients"][k]
            assert g["reference_resolved"] == (g["reference_signal_to_mc_se"] >= 5 and g["reference_split_alignment"]["cosine"] >= .95) == True
            a = g["alignment"]
            close(a["relative_error"]**2, 1+a["norm_ratio"]**2-2*a["norm_ratio"]*a["cosine"])
        for t in r["timing"]:
            close(st.mean(t["milliseconds_per_batch"]), t["mean_ms"])
            close(t["mean_ms"]/t["batch"], t["amortized_ms_per_sample"])
        d, g = r["distribution"], r["early_codec_gradients"]["encoder"]["alignment"]
        sampler_summary.append(dict(family=r["family"], nfe=r["nfe"], swd_floor=d["conditional"]["swd"]/d["analytic_floor"]["swd"],
            variance_ratio=r["moments"]["variance_ratio"], encoder=g))
    variance = 1/(4*.75*10**.8)
    euler = {}
    for n in (8, 32, 128):
        coefficient = 1.
        for i in range(n):
            t = i/n
            a = ((1+variance)*t-1)/((1-t)**2+variance*t*t)
            coefficient *= 1+a/n
        euler[str(n)] = coefficient**2/variance
    output = dict(scope="Independent portable-record/source/algebra audit; no training, checkpoint or raw-gradient tensor replay",
        exported_files_verified=len(index["files"]), local_frozen_sources_verified=len(sources),
        generators=12, ae_runs=45, main_curve_points=135, paired_discordance_records=len(pairs),
        primary=primary, primary_pairs=primary_pairs, seed_effects_by_snr=by_snr, samplers=sampler_summary,
        ideal_euler_variance_ratios=euler, nominal8_noise_variance=variance,
        input_referenced_ebno_db=10*math.log10((8/6)/(2*variance)),
        generator_training_seconds=sum(g["cost"]["train_seconds"] for g in generators.values()),
        ae_training_seconds=sum(r["ae_train_seconds"] for b in blocks.values() for r in b["rows"].values()),
        interpretation="MC standard errors are conditional on fitted codecs and are not generator-seed uncertainty")
    target = ROOT/"Journal_version/evidence/coverage_followup_independent_audit_20261013.json"
    target.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({k:output[k] for k in ("exported_files_verified", "local_frozen_sources_verified", "main_curve_points", "paired_discordance_records", "primary", "ideal_euler_variance_ratios", "input_referenced_ebno_db")}, indent=2))


if __name__ == "__main__":
    main()
