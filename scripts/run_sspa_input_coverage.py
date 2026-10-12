"""Fixed input-law intervention on the completed SSPA application pilot."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.fast_channels import FastConfig, FastTrainer, FastImplant, physical_channel
from conditional_drifting.baselines.fast_evaluation import codec, train_codec, evaluate_codec, distribution_metrics, write_json
from conditional_drifting.baselines.official_sources import load_sources, sha256
from conditional_drifting.sspa_trajectory import isolated_rng
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from scripts.run_fast_channel_application import gaussian_law
from scripts.run_fast_channel_comparison import train, ensure_manifest
from scripts.run_local_gradient_fidelity import normalized_codebook, expected_loss


class CoverageTrainer(FastTrainer):
    """Narrow fork of the Gaussian/Sinkhorn updates; only anchor sampling changes."""

    def __init__(self, cfg, device, official, wflow, law="broad"):
        if cfg.channel != "SSPA" or cfg.family not in ("gaussian", "sinkhorn") or law not in ("normal", "broad"):
            raise ValueError("Coverage experiment supports Gaussian/Sinkhorn SSPA only")
        super().__init__(cfg, device, official, wflow)
        self.law = law
        self.coverage_rng = torch.Generator(device=device).manual_seed(cfg.seed+2000000)
        self.broad_anchors = 0
        self.radius_max = math.sqrt(self.spec["message_dim"]*self.n)

    def inputs(self, anchors):
        x = torch.randn(anchors, self.n, device=self.device, generator=self.condition_rng)
        if self.law == "broad":
            u = torch.rand(anchors, 2, device=self.device, generator=self.coverage_rng)
            mask = u[:, :1] < .25
            radial = x/x.norm(dim=1, keepdim=True).clamp_min(1e-12)*(u[:, 1:]*self.radius_max)
            x = torch.where(mask, radial, x)
            self.broad_anchors += mask.sum().item()
        return x

    def step(self):
        c = self.cfg
        anchors = c.batch_size//c.samples_per_input if c.family == "sinkhorn" else c.batch_size
        x = self.inputs(anchors)
        if c.family == "sinkhorn":
            x = x.repeat_interleave(c.samples_per_input, 0)
        y = physical_channel(c.channel, x)
        if c.family == "gaussian":
            loss = self.model.loss(x, y)
        else:
            self.optimizer.param_groups[0]["lr"] = 1e-6+(c.learning_rate-1e-6)*min(self.update/5000, 1.)
            g = self.model(torch.randn(len(x), self.latent_dim, device=self.device), condition=x)
            with torch.no_grad():
                r = self.model(torch.randn(len(x), self.latent_dim, device=self.device), condition=x)
            g, p, r = [v.reshape(anchors, c.samples_per_input, self.n) for v in (g, y, r)]
            losses, _ = self.wflow.drift_loss_ot(g, p, r, R_list=(c.sinkhorn_r,),
                sinkhorn_num_iter=c.sinkhorn_iterations, disable_diag_mask=True, use_quadratic_cost=True)
            loss = losses.mean()
            self.counts["training_generated_outputs"] += 2*len(x)
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2. if c.family == "sinkhorn" else 1., error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for target, value in zip(self.ema.parameters(), self.model.parameters()):
                target.lerp_(value, 1-c.ema_decay)
        self.counts["generator_updates"] += 1
        self.counts["oracle_outputs"] += len(x)
        self.counts["input_anchors"] += anchors
        self.update += 1
        info = dict(gradient_norm=norm.item(), update=self.update, loss=loss.detach().item())
        if not math.isfinite(info["loss"]):
            raise FloatingPointError("Nonfinite loss")
        return info

    def state(self):
        return dict(super().state(), coverage=dict(law=self.law, rng=self.coverage_rng.get_state(),
            broad_anchors=self.broad_anchors, radius_max=self.radius_max))

    def restore(self, state):
        coverage = state["coverage"]
        if coverage["law"] != self.law or coverage["radius_max"] != self.radius_max:
            raise ValueError("Input-law mismatch")
        super().restore(state)
        self.coverage_rng.set_state(coverage["rng"].cpu())
        self.broad_anchors = coverage["broad_anchors"]


def load_codec(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    encoder, decoder, messages = codec("SSPA", device, state["seed"])
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    return encoder, decoder, messages, state["seconds"]


@torch.no_grad()
def moments_at(sampler, inputs, samples, seed):
    with isolated_rng(seed):
        mean, variance = gaussian_law("SSPA", inputs)
        repeated = inputs.repeat_interleave(samples, 0)
        cloud = torch.cat([sampler(x) for x in repeated.split(2048)]).reshape(len(inputs), samples, -1)
        estimate = cloud.mean(1)
        error = (estimate-mean).square().mean(1).sqrt()
        return dict(mean_rmse=error.square().mean().sqrt().item(), mean_rmse_per_message=error.tolist(),
            variance_ratio=cloud.var(1).mean().item()/variance,
            mc_mean_rmse_floor=math.sqrt(variance/samples),
            mean_iq_amplitude_max=estimate.reshape(len(inputs), -1, 2).norm(dim=-1).max().item(),
            input_norms=inputs.norm(dim=1).tolist())


def generator_evaluation(implant, historical_codes, args, settings):
    result = dict(random_inputs=distribution_metrics("SSPA", implant, args.device, 979101,
        settings["eval_size"], args.anchors, args.anchor_samples, settings["projections"], 2048), radial=[])
    with isolated_rng(979102):
        direction = torch.randn(args.anchors, 8, device=args.device)
        direction /= direction.norm(dim=1, keepdim=True)
    for radius in (math.sqrt(8), 6., 10., 16., math.sqrt(512)):
        result["radial"].append(dict(radius=radius,
            **moments_at(implant, direction*radius, args.anchor_samples, 979103)))
    result["old_gaussian_codebook"] = moments_at(implant, historical_codes, args.anchor_samples, 979104)
    return result


def application_result(implant, trained, seed, args, settings):
    e, d, m, seconds = trained
    codes = normalized_codebook(e, m).detach()
    result = dict(ae_seed=seed, ae_train_seconds=seconds, codebook=codes.tolist(),
        codebook_norms=codes.norm(dim=1).tolist(), mean_squared_norm=codes.square().sum(1).mean().item(),
        analytic_curve=evaluate_codec("SSPA", e, d, m, settings["ae_eval_samples"],
            settings["eval_batch"], seed+200000, [-4, -2, 0]))
    with isolated_rng(seed+300000):
        _, result["surrogate_nominal"] = expected_loss(codes, d, implant,
            settings["ae_eval_samples"], settings["eval_batch"], seed+300000, gradient=False)
    result["own_moments"] = moments_at(implant, codes, args.anchor_samples, 979105)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=ROOT/"results/fast_sspa_application_20261012")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-dir", type=Path, default=ROOT/"results/official_fast_sources_20261012")
    p.add_argument("--wflow-source", type=Path, default=ROOT/"results/official_sources_20261010/W-Flow")
    p.add_argument("--ae-seeds", default="209111,209112,209113")
    p.add_argument("--anchors", type=int, default=64)
    p.add_argument("--anchor-samples", type=int, default=256)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    seeds = list(map(int, args.ae_seeds.split(",")))
    if not seeds or len(set(seeds)) != len(seeds) or args.anchors < 1 or args.anchor_samples < 2:
        raise ValueError("Invalid evaluation settings")
    torch.set_num_threads(2)
    parent = json.loads((args.parent/"manifest.json").read_text())
    settings = parent["settings"]
    if settings["channel"] != "SSPA":
        raise ValueError("Expected SSPA parent")
    for name, digest in parent["adapter_files"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Parent source changed: {name}")
    official = load_sources(args.source_dir)
    wflow = load_official(args.wflow_source)
    if official.records != parent["sources"] or parent["wflow"] != dict(commit=OFFICIAL_COMMIT,
            sha256=sha256(args.wflow_source/"drift_loss_ot.py")):
        raise ValueError("Parent upstream source mismatch")
    files = [args.parent/"manifest.json", args.parent/"results.json", args.parent/"analytic/final.pt"]
    files += [args.parent/family/name for family in ("gaussian", "sinkhorn") for name in ("generator.pt", "codec.pt")]
    manifest = dict(parent_files={str(f.resolve()): sha256(f) for f in files}, parent=parent,
        script_sha256=sha256(Path(__file__)), ae_seeds=seeds, anchors=args.anchors, anchor_samples=args.anchor_samples,
        law="0.75 N(0,I8) + 0.25 uniform direction times Uniform(0,sqrt(64*8))",
        selection="Fixed final EMA; no outcome-based stopping, tuning or codeword clipping",
        runtime=dict(device=args.device, torch=str(torch.__version__), cuda=torch.version.cuda,
            hardware=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"))
    ensure_manifest(args.out_dir/"manifest.json", manifest, args.resume)
    data = json.loads((args.parent/"results.json").read_text())
    inherited = {r["method"]: r for r in data["rows"]}
    first_seed = settings["seed"]+200000
    old_codec = load_codec(args.parent/"gaussian/codec.pt", args.device)
    historical_codes = normalized_codebook(old_codec[0], old_codec[2]).detach()
    rows, generators = [], []
    variants = ("analytic", "gaussian_normal", "gaussian_broad", "sinkhorn_normal", "sinkhorn_broad")
    for variant in variants:
        if variant == "analytic":
            implant = lambda x: physical_channel("SSPA", x)
        else:
            family, law = variant.split("_")
            cfg = FastConfig(**inherited[family]["cost"]["config"])
            if law == "normal":
                state = torch.load(args.parent/family/"generator.pt", map_location="cpu", weights_only=False)["trainer"]
                if state["config"] != asdict(cfg) or state["update"] != cfg.updates:
                    raise ValueError("Invalid parent generator")
                trainer = FastTrainer(cfg, args.device, official, wflow)
                trainer.model.load_state_dict(state["model"])
                trainer.ema.load_state_dict(state["ema"])
                cost = deepcopy(inherited[family]["cost"])
            else:
                trainer = CoverageTrainer(cfg, args.device, official, wflow)
                train(trainer, args.out_dir/variant/"generator.pt", settings["save_every"], args.resume)
                cost = dict(config=asdict(cfg), counts=trainer.counts, train_seconds=trainer.train_seconds,
                    broad_anchors=trainer.broad_anchors, radius_max=trainer.radius_max)
                if trainer.counts["oracle_outputs"] != cfg.updates*cfg.batch_size:
                    raise ValueError("Query budget mismatch")
            implant = FastImplant(trainer, 1)
            path = args.out_dir/variant/"generator_evaluation.json"
            if path.exists() and args.resume:
                gen = json.loads(path.read_text())
            else:
                gen = dict(variant=variant, cost=cost,
                    evaluation=generator_evaluation(implant, historical_codes, args, settings))
                write_json(path, gen)
            generators.append(gen)
        for seed in seeds:
            path = args.out_dir/variant/f"ae{seed}"
            result_path = path/"result.json"
            if result_path.exists() and args.resume:
                rows.append(json.loads(result_path.read_text()))
                continue
            reuse = seed == first_seed and not variant.endswith("_broad")
            if reuse:
                parent_path = args.parent/("analytic/final.pt" if variant == "analytic" else f"{variant.split('_')[0]}/codec.pt")
                trained = load_codec(parent_path, args.device)
            else:
                trained = train_codec(path/"codec.pt", "SSPA", implant, args.device, seed,
                    settings["ae_updates"], settings["ae_batch"], settings["save_every"], args.resume)
            result = dict(variant=variant, reused_parent_codec=reuse,
                **application_result(implant, trained, seed, args, settings))
            if reuse:
                original = inherited["analytic" if variant == "analytic" else variant.split("_")[0]]
                for old, new in zip(original["analytic_channel_evaluation"], result["analytic_curve"]):
                    if (old["errors"], old["bit_errors"], old["trials"]) != (new["errors"], new["bit_errors"], new["trials"]):
                        raise ValueError("Inherited codec evaluation did not reproduce its error counts")
            write_json(result_path, result)
            rows.append(result)
            write_json(args.out_dir/"progress.json", dict(completed_ae_rows=len(rows), expected_ae_rows=len(variants)*len(seeds)))
            print(f"[done] {variant} AE={seed} SER={result['analytic_curve'][-1]['ser']:.6g} max_norm={max(result['codebook_norms']):.4f}", flush=True)
    write_json(args.out_dir/"results.json", dict(rows=rows, generators=generators, manifest_sha256=sha256(args.out_dir/"manifest.json")))


if __name__ == "__main__":
    main()
