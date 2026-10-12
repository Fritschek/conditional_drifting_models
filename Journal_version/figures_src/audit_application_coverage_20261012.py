"""Independently audit portable result records; no Torch, training, or checkpoints."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import tarfile


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Journal_version/evidence"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_point(point):
    n, k, b = point["trials"], point["errors"], point["bit_errors"]
    assert isinstance(k, int) and isinstance(b, int) and 0 <= k <= n
    assert k <= b <= 6*k
    assert point["ser"] == k/n and point["ber"] == b/(6*n)


def main():
    app = BASE / "fast_sspa_application_20261012"
    cov = BASE / "sspa_input_coverage_20261012"
    archives = {}
    for stem in ("fast_sspa_application", "sspa_input_coverage"):
        path = BASE / f"{stem}_evidence_20261012.tar.gz"
        assert digest(path) == (BASE / f"{stem}_evidence_20261012.sha256").read_text().split()[0]
        with tarfile.open(path) as archive:
            members = [m for m in archive if m.isfile()]
            for m in members:
                assert not Path(m.name).is_absolute() and ".." not in Path(m.name).parts
                assert archive.extractfile(m).read() == (ROOT / m.name).read_bytes(), m.name
            archives[stem] = dict(sha256=digest(path), byte_identical_files=len(members),
                                 checkpoint_files=sum(m.name.endswith(".pt") for m in members))
    a, c, manifest = read(app/"results.json"), read(cov/"results.json"), read(app/"manifest.json")
    for name, sha in manifest["adapter_files"].items():
        assert digest(ROOT/name) == sha, name
    cm = read(cov/"manifest.json")
    assert cm["parent"] == manifest
    assert digest(ROOT/"scripts/run_sspa_input_coverage.py") == cm["script_sha256"]
    assert c["manifest_sha256"] == digest(cov/"manifest.json")
    assert read(cov/"audit.json")["result_sha256"] == digest(cov/"results.json")
    aliases = {"codebook_check": "own_codebook_check.json", "support_check": "support_check.json"}
    for name, sha in read(app/"source_hashes.json").items():
        assert digest(app/aliases.get(name, name)) == sha
    own, support = read(app/"own_codebook_check.json"), read(app/"support_check.json")
    assert own["source_results_sha256"] == digest(app/"results.json")
    assert support["own_codebook_check_sha256"] == digest(app/"own_codebook_check.json")
    for data, script in ((own, "check_fast_application_codebooks.py"),
                         (support, "check_fast_application_support.py")):
        assert data["script_sha256"] == digest(ROOT/"scripts"/script)
    app_rows = {r["method"]: r for r in a["rows"]}
    assert len(app_rows) == len(a["rows"]) == 7
    cost_rows = {}
    for method, row in app_rows.items():
        for point in row["analytic_channel_evaluation"]:
            check_point(point)
        if method == "analytic":
            continue
        cost = row["cost"]
        assert cost["oracle_outputs"] == 40960000
        assert cost["counts"]["oracle_outputs"] + cost.get("teacher_counts", {}).get("oracle_outputs", 0) == 40960000
        for stage, gradients in row["gradients"].items():
            for space in ("code", "encoder"):
                g = gradients[space]
                resolved = g["reference_signal_to_mc_se"] >= 5 and g["reference_split_alignment"]["cosine"] >= .95
                assert g["reference_resolved"] == resolved == (stage == "early")
        for timing in row["timing"]:
            assert math.isclose(statistics.mean(timing["milliseconds_per_batch"]), timing["mean_ms"], rel_tol=1e-12)
        cost_rows[method] = dict(generator_seconds=cost["train_seconds"], ae_seconds=row["ae_train_seconds"],
            one_generator_plus_one_ae_seconds=cost["train_seconds"]+row["ae_train_seconds"])
    own_rows = {r["method"]: r for r in own["rows"]}
    localization = {}
    for r in support["rows"]:
        prior = own_rows[r["method"]]
        check_point(prior["analytic"])
        check_point(prior["surrogate"])
        errors, norms = r["analytic_errors_per_message"], r["input_norms"]
        assert sum(errors) == prior["analytic"]["errors"]
        for norm, tail in zip(norms, r["radial_tail_probabilities"]):
            # The recorded routine squares the float32 norm before conversion.
            z = struct.unpack("f", struct.pack("f", norm*norm))[0]/2
            expected = math.exp(-z)*sum(z**j/math.factorial(j) for j in range(4))
            assert math.isclose(tail, expected, rel_tol=1e-10)
        largest = sorted(range(64), key=lambda i: norms[i], reverse=True)[:2]
        localization[r["method"]] = dict(total_errors=sum(errors),
            errors_two_largest_norms=sum(errors[i] for i in largest), max_norm=max(norms))
    rows = {(r["variant"], r["ae_seed"]): r for r in c["rows"]}
    variants = ("analytic", "gaussian_normal", "gaussian_broad", "sinkhorn_normal", "sinkhorn_broad")
    assert set(rows) == {(v, s) for v in variants for s in cm["ae_seeds"]}
    assert len(rows) == len(c["rows"]) == 15
    assert sum(r["reused_parent_codec"] for r in rows.values()) == 3
    for (variant, seed), row in rows.items():
        codes = row["codebook"]
        norm2 = [sum(v*v for v in code) for code in codes]
        assert abs(statistics.mean(norm2)-8) < 1e-5
        # Actual normalization subtracts one scalar grand mean, not a vector mean.
        assert abs(statistics.mean(v for code in codes for v in code)) < 1e-6
        assert max(norm2) <= 512+1e-5
        for point in row["analytic_curve"] + [row["surrogate_nominal"]]:
            check_point(point)
        if row["reused_parent_codec"]:
            method = variant.split("_")[0]
            assert row["analytic_curve"] == app_rows[method]["analytic_channel_evaluation"]
    generators = {r["variant"]: r for r in c["generators"]}
    paired = {}
    for family in ("gaussian", "sinkhorn"):
        normal, broad = (generators[family+"_"+law]["cost"] for law in ("normal", "broad"))
        assert normal["config"] == broad["config"] and normal["counts"] == broad["counts"]
        assert normal["counts"]["oracle_outputs"] == 40960000
        paired[family] = []
        for seed in cm["ae_seeds"]:
            n, b = rows[family+"_normal", seed], rows[family+"_broad", seed]
            ns, bs = n["analytic_curve"][-1]["ser"], b["analytic_curve"][-1]["ser"]
            paired[family].append(dict(ae_seed=seed, normal_ser_8db=ns, broad_ser_8db=bs,
                normal_divided_by_broad=ns/bs, normal_errors=n["analytic_curve"][-1]["errors"],
                broad_errors=b["analytic_curve"][-1]["errors"]))
    # The CSV exports must contain precisely these numerical records.
    for path, expected, keys in (
        (app/"curves.csv", [dict(method=r["method"], **p) for r in a["rows"] for p in r["analytic_channel_evaluation"]], ("method",)),
        (cov/"curves.csv", [dict(variant=r["variant"], ae_seed=r["ae_seed"], **p) for r in c["rows"] for p in r["analytic_curve"]], ("variant",))):
        actual = list(csv.DictReader(path.open()))
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            assert set(x) == set(y)
            for k in y:
                assert (x[k] == y[k]) if k in keys else (float(x[k]) == y[k])
    result = dict(scope="Portable records only; no GPU execution, checkpoint tensors, or external upstream source revalidation",
        archives=archives, local_adapter_hashes_verified=len(manifest["adapter_files"]),
        application_rows=7, application_curve_points=21, coverage_rows=15, coverage_curve_points=45,
        inherited_curve_records_identical=9, gradient_reference_flags_verified=24,
        query_energy_ratio=(.75*8+.25*512/3)/8, actual_codebook_radius_bound=math.sqrt(512),
        normalization="Scalar grand-mean subtraction followed by unit mean square over all coordinates",
        maximum_absolute_coordinate_mean=max(abs(statistics.mean(code[j] for code in r['codebook']))
            for r in rows.values() for j in range(8)),
        paired_8db=paired, error_localization=localization, costs=cost_rows,
        checkpoint_level_replay="Unavailable in this checkout: compact archives contain no checkpoints",
        earlier_stability_evidence="Oct10-11 result notes and source inspected; corresponding new raw bundles absent")
    out = BASE/"application_coverage_independent_audit_20261012.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
