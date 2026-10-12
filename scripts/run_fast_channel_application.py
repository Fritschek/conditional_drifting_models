"""Matched-query AWGN/SSPA comparison with counted distillation and AE utility."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.fast_channels import (FastConfig, FastTrainer, FastImplant,
    channel_spec, physical_channel, parameter_count)
from conditional_drifting.baselines.fast_evaluation import (save, write_json, sync, distribution_metrics,
    train_codec, evaluate_codec, timings)
from conditional_drifting.baselines.official_sources import load_sources, definitions, sha256
from conditional_drifting.channels import sspa
from conditional_drifting.sspa_trajectory import isolated_rng
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from scripts.run_fast_channel_comparison import source_files, ensure_manifest, train
from scripts.run_consistency_warmstart_development import loss_step
from scripts.run_local_gradient_fidelity import normalized_codebook, expected_loss, encoder_gradient, alignment

NFES = dict(gaussian=1, wgan=1, diffusion=20, flow=8, sinkhorn=1, consistency_distilled=1)


def gaussian_law(channel, x):
    spec = channel_spec(channel)
    if channel == "AWGN":
        return x, spec["noise_std"]**2
    if channel == "SSPA":
        return sspa(x, 0., x.device), spec["noise_std"]**2/2
    raise ValueError("Closed-form moments are implemented only for AWGN and SSPA")


@torch.no_grad()
def moments(channel, sampler, device, seed, anchors, samples):
    with isolated_rng(seed):
        x = torch.randn(anchors, channel_spec(channel)["n"], device=device)
        mean, variance = gaussian_law(channel, x)
        repeated = x.repeat_interleave(samples, 0)
        y = torch.cat([sampler(v) for v in repeated.split(1024)]).reshape(anchors, samples, -1)
        return dict(mean_rmse=(y.mean(1)-mean).square().mean().sqrt().item(),
            variance_ratio=y.var(1).mean().item()/variance, analytic_variance=variance,
            mean_mc_rmse_floor=math.sqrt(variance/samples))


def edm_stage(trainer, kind, path, official, sampler, args, teacher=None):
    trace = []
    if path.exists():
        if not args.resume:
            raise FileExistsError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["kind"] != kind:
            raise ValueError("Distillation checkpoint stage mismatch")
        trainer.restore(checkpoint["trainer"])
        trace = checkpoint["trace"]
    denoiser = official.cm.KarrasDenoiser(sigma_data=trainer.cfg.consistency_sigma_data,
        distillation=False, loss_norm="l2", weight_schedule="karras")
    while trainer.update < trainer.cfg.updates:
        sync(trainer.device)
        start = time.perf_counter()
        row = loss_step(trainer, kind, denoiser, sampler, teacher)
        sync(trainer.device)
        trainer.train_seconds += time.perf_counter()-start
        if trainer.update % 100 == 0:
            trace.append(dict(**row, train_seconds=trainer.train_seconds))
        if trainer.update % args.save_every == 0 or trainer.update == trainer.cfg.updates:
            save(path, dict(kind=kind, trainer=trainer.state(), trace=trace))
            print(f"[train] {trainer.cfg.channel} {kind} update={trainer.update} loss={row['loss']:.5g}", flush=True)


def build_generator(name, args, official, wflow, directory):
    cfg = FastConfig(channel=args.channel, family="consistency" if name == "consistency_distilled" else name,
        seed=args.seed, batch_size=args.batch_size, updates=args.updates, parameters=args.parameters)
    if name != "consistency_distilled":
        trainer = FastTrainer(cfg, args.device, official, wflow)
        train(trainer, directory / "generator.pt", args.save_every, args.resume)
        cost = dict(train_seconds=trainer.train_seconds, oracle_outputs=trainer.counts["oracle_outputs"],
            counts=trainer.counts, outer_updates=trainer.update,
            generator_updates=trainer.counts["generator_updates"],
            critic_parameters=0 if trainer.critic is None else parameter_count(trainer.critic))
    else:
        cfg = replace(cfg, updates=args.updates//2)
        namespace = dict(vars(official.cm))
        definitions(args.source_dir / "consistency_models/cm/resample.py", ("LogNormalSampler",), namespace)
        sigma_sampler = namespace["LogNormalSampler"]()
        edm = FastTrainer(cfg, args.device, official)
        edm_stage(edm, "edm", directory / "edm.pt", official, sigma_sampler, args)
        teacher = deepcopy(edm.ema).eval().requires_grad_(False)
        trainer = FastTrainer(replace(cfg, seed=cfg.seed+10000), args.device, official)
        for model in (trainer.model, trainer.ema, trainer.target):
            model.load_state_dict(teacher.state_dict())
        edm_stage(trainer, "cd", directory / "generator.pt", official, sigma_sampler, args, teacher)
        cost = dict(train_seconds=edm.train_seconds+trainer.train_seconds,
            oracle_outputs=edm.counts["oracle_outputs"]+trainer.counts["oracle_outputs"],
            counts=trainer.counts, teacher_counts=edm.counts,
            teacher_train_seconds=edm.train_seconds, student_train_seconds=trainer.train_seconds,
            teacher_parameters=parameter_count(teacher), teacher_pretraining_included=True)
    cost.update(inference_parameters=parameter_count(trainer.model), nfe=NFES[name],
        config=asdict(trainer.cfg), ema_parameters=parameter_count(trainer.ema),
        target_parameters=0 if trainer.target is None else parameter_count(trainer.target))
    if cost["oracle_outputs"] != args.updates*args.batch_size:
        raise ValueError("Oracle exposure budget mismatch")
    return FastImplant(trainer, NFES[name]), cost


def reference_gradients(encoder, decoder, messages, args, seed):
    codes = normalized_codebook(encoder, messages).detach()
    gs, es, losses = [], [], []
    with isolated_rng(seed):
        for index in range(args.gradient_repeats):
            g, loss = expected_loss(codes, decoder, lambda x: physical_channel(args.channel, x),
                args.gradient_samples, args.eval_batch, seed+index)
            gs.append(g)
            es.append(encoder_gradient(encoder, messages, g))
            losses.append(loss)
    def summarize(values):
        values = torch.stack(values)
        mean = values.mean(0)
        se_norm = ((values-mean).square().sum()/(len(values)*(len(values)-1))).sqrt()
        snr = (mean.norm()/se_norm).item() if se_norm > 0 else None
        split = alignment(values[:len(values)//2].mean(0), values[len(values)//2:].mean(0))
        return dict(mean=mean, repeats=values, mc_se_norm=se_norm.item(), signal_to_mc_se=snr,
            split_alignment=split, resolved=snr is not None and snr >= 5 and
                split["cosine"] is not None and split["cosine"] >= .95)
    return dict(code=summarize(gs), encoder=summarize(es), analytic_losses=losses)


def compare_gradients(sampler, codec, reference, args, seed):
    encoder, decoder, messages, _ = codec
    codes = normalized_codebook(encoder, messages).detach()
    gs, es, losses = [], [], []
    with isolated_rng(seed):
        for index in range(args.surrogate_gradient_repeats):
            g, loss = expected_loss(codes, decoder, sampler, args.gradient_samples, args.eval_batch, seed+index)
            gs.append(g)
            es.append(encoder_gradient(encoder, messages, g))
            losses.append(loss)
    def summarize(values, ref):
        vectors = torch.stack(values)
        mean = vectors.mean(0)
        se = ((vectors-mean).square().sum()/(len(vectors)*(len(vectors)-1))).sqrt().item()
        return dict(alignment=alignment(mean, ref["mean"]), per_repeat_alignment=[alignment(v, ref["mean"]) for v in values],
            surrogate_mc_se_norm=se, reference_mc_se_norm=ref["mc_se_norm"],
            reference_signal_to_mc_se=ref["signal_to_mc_se"], reference_split_alignment=ref["split_alignment"],
            reference_resolved=ref["resolved"])
    return dict(code=summarize(gs, reference["code"]), encoder=summarize(es, reference["encoder"]),
        analytic_losses=reference["analytic_losses"], surrogate_losses=losses,
        samples_per_message=args.gradient_samples, reference_repeats=args.gradient_repeats,
        surrogate_repeats=args.surrogate_gradient_repeats,
        qualification="Resolution flag is a Monte Carlo heuristic, not a population guarantee")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=ROOT / "results/official_fast_sources_20261012")
    p.add_argument("--wflow-source", type=Path, default=ROOT / "results/official_sources_20261010/W-Flow")
    p.add_argument("--channel", choices=("AWGN", "SSPA"), default="SSPA")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=9111)
    p.add_argument("--updates", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--parameters", type=int, default=60000)
    p.add_argument("--ae-updates", type=int, default=20000)
    p.add_argument("--early-updates", type=int, default=2000)
    p.add_argument("--ae-batch", type=int, default=500)
    p.add_argument("--ae-eval-samples", type=int, default=16384)
    p.add_argument("--eval-batch", type=int, default=2048)
    p.add_argument("--eval-size", type=int, default=16384)
    p.add_argument("--anchors", type=int, default=64)
    p.add_argument("--anchor-samples", type=int, default=256)
    p.add_argument("--projections", type=int, default=128)
    p.add_argument("--gradient-samples", type=int, default=8192)
    p.add_argument("--gradient-repeats", type=int, default=4)
    p.add_argument("--surrogate-gradient-repeats", type=int, default=2)
    p.add_argument("--timing-repeats", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.updates <= 0 or args.updates % 2 or args.batch_size % 64:
        raise ValueError("Use an even update budget and batch divisible by 64")
    if not 0 < args.early_updates < args.ae_updates:
        raise ValueError("Early codec must precede the full AE budget")
    if min(args.gradient_repeats, args.surrogate_gradient_repeats, args.anchor_samples) < 2:
        raise ValueError("At least two samples/repeats required for uncertainty checks")
    if min(args.batch_size, args.parameters, args.ae_batch, args.ae_eval_samples, args.eval_batch,
           args.eval_size, args.anchors, args.projections, args.gradient_samples, args.timing_repeats, args.save_every) < 1:
        raise ValueError("Positive budgets and intervals required")
    torch.set_num_threads(2)
    official = load_sources(args.source_dir)
    wflow = load_official(args.wflow_source)
    hashes = source_files()
    for name in (Path(__file__).name, "run_consistency_warmstart_development.py", "run_fast_baseline_development.py"):
        hashes["scripts/"+name] = sha256(ROOT / "scripts" / name)
    settings = {k: v for k, v in vars(args).items() if k not in ("out_dir", "source_dir", "wflow_source", "resume")}
    manifest = dict(schema=1, settings=settings, channel_spec=channel_spec(args.channel),
        sampler_nfes=NFES, sources=official.records, adapter_files=hashes,
        wflow=dict(commit=OFFICIAL_COMMIT, sha256=sha256(args.wflow_source / "drift_loss_ot.py")),
        selection="Final EMA checkpoints only; no SER/gradient-based selection or retuning",
        protocol="Matched oracle output exposure, approximately matched inference capacity, measured nonmatched compute",
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda, threads=2,
            device=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu"))
    ensure_manifest(args.out_dir / "manifest.json", manifest, args.resume)
    codecs, references = {}, {}
    analytic = lambda x: physical_channel(args.channel, x)
    for label, updates in (("early", args.early_updates), ("final", args.ae_updates)):
        path = args.out_dir / "analytic" / f"{label}.pt"
        if label == "final" and not path.exists():
            save(path, torch.load(args.out_dir / "analytic/early.pt", map_location="cpu", weights_only=False))
        codecs[label] = train_codec(path, args.channel, analytic, args.device, args.seed+200000,
            updates, args.ae_batch, args.save_every, args.resume or label == "final")
        ref_path = args.out_dir / "analytic" / f"{label}_gradients.pt"
        if ref_path.exists() and args.resume:
            references[label] = torch.load(ref_path, map_location=args.device, weights_only=False)
        else:
            references[label] = reference_gradients(*codecs[label][:3], args, args.seed+800000+updates)
            save(ref_path, references[label])
        print(f"[reference] {label} encoder_gradient_resolved={references[label]['encoder']['resolved']}", flush=True)
    reference_result = dict(method="analytic", ae_train_seconds=codecs["final"][3],
        analytic_channel_evaluation=evaluate_codec(args.channel, *codecs["final"][:3], args.ae_eval_samples,
            args.eval_batch, args.seed+400000, [-4, -2, 0]))
    write_json(args.out_dir / "analytic/result.json", reference_result)
    results = [reference_result]
    for name, nfe in NFES.items():
        directory = args.out_dir / name
        result_path = directory / "result.json"
        if result_path.exists() and args.resume:
            results.append(json.loads(result_path.read_text()))
            continue
        implant, cost = build_generator(name, args, official, wflow, directory)
        evaluation_path = directory / "generator_evaluation.json"
        if evaluation_path.exists() and args.resume:
            result = json.loads(evaluation_path.read_text())
        else:
            result = evaluate_generator(name, cost, implant, codecs, references, args)
            write_json(evaluation_path, result)
        encoder, decoder, messages, seconds = train_codec(directory / "codec.pt", args.channel, implant,
            args.device, args.seed+200000, args.ae_updates, args.ae_batch, args.save_every, args.resume)
        result["ae_train_seconds"] = seconds
        result["analytic_channel_evaluation"] = evaluate_codec(args.channel, encoder, decoder, messages,
            args.ae_eval_samples, args.eval_batch, args.seed+400000, [-4, -2, 0])
        write_json(result_path, result)
        results.append(result)
        print(f"[done] {name} nfe={nfe} nominal_SER={result['analytic_channel_evaluation'][-1]['ser']:.6g}", flush=True)
    write_json(args.out_dir / "results.json", dict(channel=args.channel, seed=args.seed, rows=results))


def evaluate_generator(name, cost, implant, codecs, references, args):
    result = dict(method=name, cost=cost,
            distribution=distribution_metrics(args.channel, implant, args.device, args.seed+300000,
                args.eval_size, args.anchors, args.anchor_samples, args.projections, args.eval_batch),
            moments=moments(args.channel, implant, args.device, args.seed+500000, args.anchors, args.anchor_samples),
            timing=timings(implant, channel_spec(args.channel)["n"], args.device, args.seed+600000,
                [1, args.ae_batch], args.timing_repeats), gradients={}, codebook_distributions={})
    for label, updates in (("early", args.early_updates), ("final", args.ae_updates)):
        result["gradients"][label] = compare_gradients(implant, codecs[label], references[label],
            args, args.seed+850000+updates)
        result["codebook_distributions"][label] = distribution_metrics(args.channel, implant, args.device,
            args.seed+700000+updates, args.eval_size, args.anchors, args.anchor_samples, args.projections,
            args.eval_batch, codes=normalized_codebook(codecs[label][0], codecs[label][2]))
    return result


if __name__ == "__main__":
    main()
