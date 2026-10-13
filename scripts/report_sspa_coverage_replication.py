"""Audit the replication grid and summarize paired effects by generator seed."""

import argparse
import csv
import io
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.baselines.fast_evaluation import write_json
from conditional_drifting.baselines.official_sources import sha256
from scripts.coverage_followup_common import read, VARIANTS


def summarize(suite):
    manifest = read(suite / "manifest.json")
    results = read(suite / "results.json")
    if results["manifest_sha256"] != sha256(suite / "manifest.json"):
        raise ValueError("Manifest hash mismatch")
    for name, digest in manifest["sources"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen child source changed: {name}")
    plan = manifest["plan"]
    expected = {(s, a) for s in plan["generator_seeds"] for a in plan["ae_seeds"][str(s)]}
    blocks = {(b["generator_seed"], b["ae_seed"]): b for b in results["blocks"]}
    if set(blocks) != expected or len(blocks) != len(results["blocks"]):
        raise ValueError("Incomplete or duplicate AE blocks")
    generators = {(g["generator_seed"], g["variant"]): g for g in results["generators"]}
    if set(generators) != {(s, v) for s in plan["generator_seeds"] for v in VARIANTS[1:]} or len(generators) != len(results["generators"]):
        raise ValueError("Incomplete or duplicate generators")
    for (seed, variant), g in generators.items():
        checkpoint = suite / f"seed{seed}" / variant / "generator.pt"
        if sha256(checkpoint) != g["checkpoint_sha256"]:
            raise ValueError("Generator checkpoint hash mismatch")
        if g["cost"]["counts"]["oracle_outputs"] != plan["settings"]["updates"]*plan["settings"]["batch_size"]:
            raise ValueError("Exposure mismatch")
    flat, effects = [], []
    for (seed, ae), block in sorted(blocks.items()):
        if set(block["rows"]) != set(VARIANTS) or [c["ebno_db"] for c in block["curves"]] != [4., 6., 8.]:
            raise ValueError("Incomplete variants or SNR grid")
        saved = read(suite / f"seed{seed}" / f"paired_ae{ae}.json")
        if saved["result"] != block or saved["stamp"]["manifest"] != results["manifest_sha256"]:
            raise ValueError("Paired record mismatch")
        for variant, row in block["rows"].items():
            ckpt = suite / f"seed{seed}" / variant / f"ae{ae}" / "codec.pt"
            if sha256(ckpt) != saved["stamp"]["codecs"][variant]:
                raise ValueError("AE checkpoint hash mismatch")
            mean_power = sum(sum(x*x for x in code) for code in row["codebook"])/64
            if abs(mean_power-8) > 1e-5:
                raise ValueError("Codebook average power changed")
        for curve in block["curves"]:
            for variant, p in curve["variants"].items():
                n, k, bits = p["trials"], p["errors"], p["bit_errors"]
                if n != 64*plan["settings"]["ae_eval_samples"] or not 0 <= k <= n or not k <= bits <= 6*k:
                    raise ValueError("Invalid trial counts")
                if p["ser"] != k/n or p["ber"] != bits/(6*n) or sum(p["errors_per_message"]) != k or sum(p["bit_errors_per_message"]) != bits:
                    raise ValueError("SER/BER accounting mismatch")
                flat.append(dict(generator_seed=seed, ae_seed=ae, variant=variant,
                                 ebno_db=curve["ebno_db"], ser=p["ser"], ber=p["ber"], errors=k, trials=n))
            expected_pairs = {(a, b) for i, a in enumerate(VARIANTS) for b in VARIANTS[i+1:]}
            if len(curve["pairs"]) != 10 or {(p["a"], p["b"]) for p in curve["pairs"]} != expected_pairs:
                raise ValueError("Missing paired comparisons")
            for pair in curve["pairs"]:
                a, b = curve["variants"][pair["a"]], curve["variants"][pair["b"]]
                if pair["a_only_wrong"]+pair["both_wrong"] != a["errors"] or pair["b_only_wrong"]+pair["both_wrong"] != b["errors"]:
                    raise ValueError("Discordance accounting mismatch")
                if sum(pair[k] for k in ("a_only_wrong", "b_only_wrong", "both_wrong", "neither_wrong")) != a["trials"]:
                    raise ValueError("Paired trial mismatch")
        nominal = block["curves"][-1]["variants"]
        for family in ("gaussian", "sinkhorn"):
            normal, broad = nominal[family+"_normal"]["ser"], nominal[family+"_broad"]["ser"]
            effects.append(dict(generator_seed=seed, ae_seed=ae, family=family,
                normal_ser=normal, broad_ser=broad, broad_minus_normal=broad-normal))
    grouped = []
    for family in ("gaussian", "sinkhorn"):
        values = [dict(generator_seed=s, mean_paired_ser_change=statistics.mean(
            e["broad_minus_normal"] for e in effects if e["family"] == family and e["generator_seed"] == s))
            for s in plan["generator_seeds"]]
        means = [v["mean_paired_ser_change"] for v in values]
        pairs = [e for e in effects if e["family"] == family]
        normal_mean = statistics.mean(e["normal_ser"] for e in pairs)
        broad_mean = statistics.mean(e["broad_ser"] for e in pairs)
        grouped.append(dict(family=family, generator_blocks=values, mean_change=statistics.mean(means),
            between_generator_sd=statistics.stdev(means) if len(means)>1 else None,
            normal_mean_ser=normal_mean, broad_mean_ser=broad_mean,
            normal_divided_by_broad=None if broad_mean == 0 else normal_mean/broad_mean,
            numerically_improved_pairs=sum(e["broad_minus_normal"] < 0 for e in pairs),
            numerically_worsened_pairs=sum(e["broad_minus_normal"] > 0 for e in pairs),
            qualification="Generator blocks are the replication units; AE repeats and MC trials are not independent generators"))
    return dict(profile=plan["profile"], generators=len(generators), ae_runs=5*len(blocks),
                curve_points=len(flat), primary=grouped, paired_effects=effects,
                aggregation_script_sha256=sha256(__file__)), flat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    args = p.parse_args()
    summary, rows = summarize(args.suite_dir)
    write_json(args.suite_dir / "summary.json", summary)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (args.suite_dir / "curves.csv").write_text(text.getvalue())
    print(f"Audited {summary['generators']} generators, {summary['ae_runs']} AEs, {summary['curve_points']} curve points ({summary['profile']}).")


if __name__ == "__main__":
    main()
