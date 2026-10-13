"""Post-hoc separation of SSPA codebook and decoder errors; no training or model selection."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.application_channels import SSPACoverageChannel
from conditional_drifting.baselines.application_generators import FastConfig, FastTrainer, FastImplant
from conditional_drifting.baselines.fast_evaluation import write_json
from conditional_drifting.baselines.official_sources import load_sources, sha256
from conditional_drifting.sspa_update_screen import load_official
from scripts.run_sspa_input_coverage import load_codec


@torch.no_grad()
def run(args):
    torch.set_num_threads(2)
    parent = json.loads((args.suite_dir/"results.json").read_text())
    manifest = json.loads((args.suite_dir/"manifest.json").read_text())
    for name, digest in manifest["sources"]["files"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Source changed: {name}")
    for name, digest in parent["checkpoint_hashes"].items():
        if sha256(args.suite_dir/name) != digest:
            raise ValueError(f"Checkpoint changed: {name}")
    channel = SSPACoverageChannel(input_law=manifest["settings"]["input_law"])
    official = load_sources(ROOT/"results/official_fast_sources_20261012")
    wflow = load_official(ROOT/"results/official_sources_20261010/W-Flow")
    blocks = []
    for block in parent["blocks"]:
        rows = []
        for name, model in block["models"].items():
            codes = torch.tensor(model["codebook"])
            means = channel.mean(codes)
            _, decoder, _, _ = load_codec(args.suite_dir/name/f"ae{block['ae_seed']}.pt", "cpu")
            rng = torch.Generator().manual_seed(block["ae_seed"]+800000)
            counts = dict(neural_errors=0, nearest_mean_errors=0, neural_only_wrong=0,
                          nearest_only_wrong=0, both_wrong=0)
            per_word = {k:torch.zeros(64,dtype=torch.long) for k in ("neural", "nearest")}
            for start in range(0, args.samples, 32):
                count = min(32, args.samples-start)
                labels = torch.arange(64).repeat_interleave(count)
                y = means.repeat_interleave(count, 0)+math.sqrt(channel.variance)*torch.randn(64*count,8,generator=rng)
                neural = decoder(y).argmax(-1) != labels
                # Equal priors and common spherical Gaussian noise make this exact ML.
                nearest = (2*y@means.T-means.square().sum(-1)).argmax(-1) != labels
                counts["neural_errors"] += neural.sum().item()
                counts["nearest_mean_errors"] += nearest.sum().item()
                counts["neural_only_wrong"] += (neural & ~nearest).sum().item()
                counts["nearest_only_wrong"] += (~neural & nearest).sum().item()
                counts["both_wrong"] += (neural & nearest).sum().item()
                per_word["neural"] += neural.reshape(64,count).sum(-1)
                per_word["nearest"] += nearest.reshape(64,count).sum(-1)
            row = dict(method=name, trials=64*args.samples, **counts,
                neural_ser=counts["neural_errors"]/(64*args.samples),
                nearest_mean_ser=counts["nearest_mean_errors"]/(64*args.samples),
                per_message={k:v.tolist() for k,v in per_word.items()})
            if name != "analytic":
                family, nfe = name.rsplit("_",1)
                state = torch.load(args.suite_dir/family/"generator.pt",map_location="cpu",weights_only=False)["trainer"]
                trainer = FastTrainer(FastConfig(**state["config"]), "cpu", official, channel, wflow)
                trainer.model.load_state_dict(state["model"])
                trainer.ema.load_state_dict(state["ema"])
                torch.manual_seed(block["ae_seed"]+810000)
                sampler = FastImplant(trainer,int(nfe))
                generated = torch.cat([sampler(x) for x in codes.repeat_interleave(args.moment_samples,0).split(1024)])
                cloud = generated.reshape(64,args.moment_samples,8)
                error = (cloud.mean(1)-means).square().mean(-1).sqrt()
                row["own_codebook_moments"] = dict(mean_rmse=error.square().mean().sqrt().item(),
                    mean_rmse_per_message=error.tolist(), variance_ratio=cloud.var(1).mean().item()/channel.variance,
                    variance_ratio_per_message=(cloud.var(1).mean(-1)/channel.variance).tolist(),
                    analytic_mean_mc_rmse_floor=math.sqrt(channel.variance/args.moment_samples))
            rows.append(row)
            print(name, row["neural_errors"], row["nearest_mean_errors"],flush=True)
        blocks.append(dict(ae_seed=block["ae_seed"],rows=rows))
    write_json(args.out,dict(parent_results_sha256=sha256(args.suite_dir/"results.json"),
        parent_manifest_sha256=sha256(args.suite_dir/"manifest.json"), script_sha256=sha256(__file__),
        physical=channel.description(), samples_per_message=args.samples, moment_samples=args.moment_samples,
        device="cpu", torch=str(torch.__version__), blocks=blocks,
        scope="Post-hoc own-codebook analysis with new CPU evaluation draws. Exact ML for the implemented equal-prior SSPA law, not a new decoder-training result or pre-optimization metric."))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir",type=Path,default=ROOT/"results/common_coverage_9301")
    p.add_argument("--out",type=Path,default=ROOT/"results/common_coverage_9301/codebook_inspection.json")
    p.add_argument("--samples",type=int,default=16384)
    p.add_argument("--moment-samples",type=int,default=512)
    args=p.parse_args()
    if args.samples < 1 or args.moment_samples < 2 or args.out.exists():
        raise ValueError("Positive counts and a new output file required")
    run(args)
