"""Post-hoc transfer checks at each surrogate-trained codec, without retraining."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer, FastImplant, physical_channel
from conditional_drifting.baselines.fast_evaluation import codec, distribution_metrics, write_json
from conditional_drifting.baselines.official_sources import load_sources, sha256
from conditional_drifting.sspa_trajectory import isolated_rng
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from scripts.run_fast_channel_application import gaussian_law
from scripts.run_local_gradient_fidelity import normalized_codebook, expected_loss


def load_implant(suite, name, row, manifest, official, wflow, device):
    checkpoint = torch.load(suite / name / "generator.pt", map_location="cpu", weights_only=False)
    state = checkpoint["trainer"]
    if state["config"] != row["cost"]["config"] or state["update"] != state["config"]["updates"]:
        raise ValueError("Incomplete or incompatible generator")
    if name == "consistency_distilled" and checkpoint.get("kind") != "cd":
        raise ValueError("Expected distilled student")
    if row["cost"]["nfe"] != manifest["sampler_nfes"][name]:
        raise ValueError("Sampler mismatch")
    cfg = FastConfig(**state["config"])
    with isolated_rng(cfg.seed):
        trainer = FastTrainer(cfg, device, official, wflow)
        # Load inference weights only: CUDA training RNG state is not CPU-portable.
        trainer.model.load_state_dict(state["model"])
        trainer.ema.load_state_dict(state["ema"])
        return FastImplant(trainer, row["cost"]["nfe"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-dir", type=Path, default=ROOT / "results/official_fast_sources_20261012")
    p.add_argument("--wflow-source", type=Path, default=ROOT / "results/official_sources_20261010/W-Flow")
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--anchor-samples", type=int, default=1024)
    args = p.parse_args()
    if args.samples < 1 or args.anchor_samples < 2:
        raise ValueError("Positive loss samples and at least two anchor samples required")
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    data = json.loads((args.suite_dir / "results.json").read_text())
    for name, digest in manifest["adapter_files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Source changed: {name}")
    official = load_sources(args.source_dir)
    if official.records != manifest["sources"]:
        raise ValueError("Official source mismatch")
    wflow = load_official(args.wflow_source)
    if manifest["wflow"] != dict(commit=OFFICIAL_COMMIT, sha256=sha256(args.wflow_source / "drift_loss_ot.py")):
        raise ValueError("W-Flow source mismatch")
    rows = []
    for row in data["rows"]:
        name = row["method"]
        if name == "analytic":
            continue
        implant = load_implant(args.suite_dir, name, row, manifest, official, wflow, args.device)
        path = args.suite_dir / name / "codec.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["update"] != manifest["settings"]["ae_updates"]:
            raise ValueError("Incomplete codec")
        encoder, decoder, messages = codec(data["channel"], args.device, state["seed"])
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        codes = normalized_codebook(encoder, messages).detach()
        seed = data["seed"]+960000
        metrics = distribution_metrics(data["channel"], implant, args.device, seed, 16384,
            len(codes), args.anchor_samples, 128, 2048, codes=codes)
        with isolated_rng(seed):
            _, analytic_loss = expected_loss(codes, decoder, lambda x: physical_channel(data["channel"], x),
                args.samples, 2048, seed+1, gradient=False)
            _, surrogate_loss = expected_loss(codes, decoder, implant, args.samples, 2048, seed+2, gradient=False)
            with torch.no_grad():
                repeated = codes.repeat_interleave(args.anchor_samples, 0)
                cloud = torch.cat([implant(x) for x in repeated.split(2048)]).reshape(len(codes), args.anchor_samples, -1)
                mean, variance = gaussian_law(data["channel"], codes)
                rmses = (cloud.mean(1)-mean).square().mean(1).sqrt()
                variance_ratio = cloud.var(1).mean().item()/variance
        radii = codes.reshape(len(codes), -1, 2).norm(dim=-1).tolist() if data["channel"] == "SSPA" else None
        result = dict(method=name, distribution=metrics, analytic=analytic_loss, surrogate=surrogate_loss,
            mean_rmse_per_message=rmses.tolist(), mean_rmse=rmses.square().mean().sqrt().item(),
            variance_ratio=variance_ratio, iq_radii_per_message=radii,
            codebook=codes.tolist(), codec_sha256=sha256(path),
            generator_sha256=sha256(args.suite_dir / name / "generator.pt"))
        rows.append(result)
        print(f"[codebook] {name}: surrogate_SER={surrogate_loss['ser']:.6g} analytic_SER={analytic_loss['ser']:.6g}", flush=True)
    write_json(args.out, dict(rows=rows, samples_per_message=args.samples, anchor_samples=args.anchor_samples,
        source_results_sha256=sha256(args.suite_dir / "results.json"), script_sha256=sha256(Path(__file__)),
        qualification="Post-hoc explanatory evaluation; no retraining or checkpoint selection"))


if __name__ == "__main__":
    main()
