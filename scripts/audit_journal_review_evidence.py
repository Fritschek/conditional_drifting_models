"""Recompute saved evidence relevant to the October 2026 TCOM reviews."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_json(path):
    return json.loads(path.read_text())


def describe(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
    }


def collect():
    evidence = {}
    archive_path = ROOT / "Journal_version/tcom_upload.zip"
    filenames = {
        "conditionwise_sinkhorn_drifting.tex": "drifting_vs_diffusion_summary.tex",
        "timing_table_cuda.tex": "timing_table_cuda.tex",
        "wflow_coding_table.tex": "wflow_coding_table.tex",
        "wflow_sspa_budget_table.tex": "wflow_sspa_budget_table.tex",
        "wflow_curve_figures.tex": "wflow_curve_figures.tex",
    }
    with zipfile.ZipFile(archive_path) as archive:
        evidence["local_upload_bundle"] = {
            "path": str(archive_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "matches_working_sources": {
                packaged: archive.read(packaged)
                == (ROOT / "Journal_version" / local).read_bytes()
                for packaged, local in filenames.items()
            },
        }

    timing = read_json(
        RESULTS / "timing_suite_local_cuda_20260601_1755/timing_suite_summary.json"
    )["results"]["cuda"]["AWGN"]["methods"]["ddim100"]
    milliseconds = timing["milliseconds_per_sample"]
    train_hours = timing["projected_full_train_hours"]
    evidence["awgn_ddim100_timing"] = {
        "train_hours": train_hours,
        "batch_amortized_milliseconds_per_sample": milliseconds,
        "implied_generation_samples": round(
            timing["projected_eval_seconds"] * 1000 / milliseconds
        ),
        "saved_total_hours": timing["projected_total_benchmark_hours"],
        "total_hours_for_one_million": train_hours + 1_000_000 * milliseconds / 3_600_000,
        "total_hours_for_ten_million": train_hours + 10_000_000 * milliseconds / 3_600_000,
    }

    evidence["sspa_generator_budgets"] = {}
    for label, suite in [
        ("full", "journal_wflow_fiber_fixed_paper_hpc_20260601_161729"),
        ("compact", "journal_wflow_sspa_budget_screen_20260602_071106"),
    ]:
        rows = []
        seed7 = None
        for path in sorted((RESULTS / suite / "fiber_sinkhorn").glob("seed*/fiber_sinkhorn_summary_seed*.json")):
            payload = read_json(path)
            channel = payload.get("channels", {}).get("SSPA")
            if channel is None:
                continue
            rows.append(channel.get("direct_swd", channel["swd"]))
            if path.parent.name == "seed7":
                seed7 = channel["config"]
        if seed7 is None:
            raise ValueError(f"Missing seed 7 SSPA config in {suite}")
        evidence["sspa_generator_budgets"][label] = {
            "suite": suite,
            "swd": describe(rows),
            "seed7_config": seed7,
            "actual_optimizer_updates": math.ceil(seed7["dataset_size"] / seed7["batch_size"]) * seed7["epochs"],
        }

    curve_root = RESULTS / "journal_wflow_curves_20260602_220608"
    table_root = RESULTS / "journal_wflow_ser_sspa_budget_screen_20260602_073353"
    evidence["sspa_ser_at_8db"] = {}
    for variant in ["analytic", "fiber_sinkhorn", "wgan", "diffusion_ddim100"]:
        values = []
        for path in sorted(curve_root.glob("curve_sspa_seed*_result.json")):
            for run in read_json(path)["runs"]:
                if run["variant"] == variant:
                    values.extend(point["ser"] for point in run["curve"] if point["ebno_db"] == 8)
        record = {"curve": describe(values)}
        if variant in {"analytic", "fiber_sinkhorn"}:
            table_values = []
            same_history = 0
            same_config = 0
            for path in sorted(table_root.glob("ser_sspa_seed*_result.json")):
                payload = read_json(path)
                seed = payload["seed"]
                table_values.extend(run["final_eval_ser"] for run in payload["runs"] if run["variant"] == variant)
                relative = Path("SSPA") / f"seed{seed}" / variant / "summary.json"
                table_run = read_json(table_root / relative)
                curve_run = read_json(curve_root / relative)
                same_history += table_run["history"] == curve_run["history"]
                same_config += table_run["config"] == curve_run["config"]
            record.update(table=describe(table_values), identical_training_histories=same_history, identical_configs=same_config)
        evidence["sspa_ser_at_8db"][variant] = record

    turbo_root = RESULTS / "turboae_long_block_hpc_20260529_165218"
    implants = [read_json(path) for path in sorted(turbo_root.glob("seed*/channel_implants/awgn2_fiber_sinkhorn/summary.json"))]
    runs = [read_json(path) for path in sorted(turbo_root.glob("seed*/checkpoint_l64/summary.json"))]
    evidence["turboae"] = {
        "surrogate_count": len(implants),
        "surrogate_dimensions": sorted({row["config"]["n"] for row in implants}),
        "surrogate_noise_stds": sorted({row["config"]["noise_std"] for row in implants}),
        "surrogate_epochs": sorted({row["config"]["epochs"] for row in implants}),
        "autoencoder_count": len(runs),
        "information_block_lengths": sorted({row["sequence_length"] for row in runs}),
        "channel_lengths": sorted({row["channel_length"] for row in runs}),
        "rates": sorted({row["rate"] for row in runs}),
        "decoder_ebno_offsets": sorted({(row["decoder_ebno_offset_low"], row["decoder_ebno_offset_high"]) for row in runs}),
    }
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    evidence = collect()
    text = json.dumps(evidence, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(args.out)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
