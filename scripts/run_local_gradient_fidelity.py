"""Frozen-codec gradient comparison using existing seed checkpoints, without training."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conditional_drifting.channels import sspa
from conditional_drifting.e2e_implants import load_implant_from_checkpoint
from conditional_drifting.metrics import conditional_anchor_swd, sliced_wasserstein_distance
from scripts.run_symbolic_awgn_benchmark import load_symbolic_checkpoint


def normalized_codebook(encoder, messages):
    codes = encoder(messages)
    codes = codes - codes.mean()
    return codes / codes.square().mean().sqrt().clamp_min(1e-12)


def physical_channel(channel, noise_std):
    if channel == "AWGN":
        return lambda x: x + noise_std * torch.randn_like(x)
    if channel == "SSPA":
        return lambda x: sspa(x, noise_std, x.device)
    raise ValueError(channel)


def expected_loss(codes, decoder, channel, samples, batch_size, seed, gradient=True):
    """Uniform messages, equally many noise draws per message, fixed noise scale."""
    torch.manual_seed(seed)
    codes = codes.detach().requires_grad_(gradient)
    m, n = codes.shape
    per_chunk = max(1, batch_size // m)
    grad = torch.zeros_like(codes)
    loss_sum = 0.0
    errors = 0
    bit_errors = 0
    bits = int(math.log2(m))
    if 2**bits != m:
        raise ValueError("Natural-label BER requires a power-of-two message alphabet.")
    with torch.set_grad_enabled(gradient):
        for start in range(0, samples, per_chunk):
            count = min(per_chunk, samples - start)
            x = codes[:, None, :].expand(m, count, n).reshape(-1, n)
            labels = torch.arange(m, device=codes.device).repeat_interleave(count)
            logits = decoder(channel(x))
            loss = F.cross_entropy(logits, labels, reduction="sum")
            loss_sum += loss.detach().double().item()
            predicted = logits.detach().argmax(-1)
            errors += (predicted != labels).sum().item()
            diff = predicted ^ labels
            bit_errors += sum(((diff >> k) & 1).sum().item() for k in range(bits))
            if gradient:
                grad += torch.autograd.grad(loss / (m * samples), codes)[0]
    return grad.detach(), {
        "ce": loss_sum / (m * samples), "ser": errors / (m * samples),
        "ber": bit_errors / (m * samples * bits), "errors": errors,
        "bit_errors": bit_errors, "trials": m * samples,
    }


def alignment(estimate, reference):
    a, b = estimate.double().flatten(), reference.double().flatten()
    na, nb = a.norm().item(), b.norm().item()
    return {
        "cosine": float(a.dot(b) / (na * nb)) if na and nb else None,
        "relative_error": float((a - b).norm() / nb) if nb else None,
        "norm_ratio": na / nb if nb else None,
    }


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "mc_se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None}


def encoder_gradient(encoder, messages, code_gradient):
    codes = normalized_codebook(encoder, messages)
    grads = torch.autograd.grad((codes * code_gradient).sum(), tuple(encoder.parameters()))
    return torch.cat([g.flatten() for g in grads]).detach()


def finite_difference_check(codes, decoder, channel, samples=128):
    # SymbolicDecoder.forward casts to float32; check its identical network in float64.
    decoder = copy.deepcopy(getattr(decoder, "net", decoder)).double()
    codes = codes.double()
    torch.manual_seed(419)
    direction = torch.randn_like(codes)
    direction /= direction.norm()
    g, _ = expected_loss(codes, decoder, channel, samples, 1024, 420)
    h = 1e-5
    _, plus = expected_loss(codes + h * direction, decoder, channel, samples, 1024, 420, False)
    _, minus = expected_loss(codes - h * direction, decoder, channel, samples, 1024, 420, False)
    fd = (plus["ce"] - minus["ce"]) / (2 * h)
    ad = (g * direction).sum().item()
    error = abs(fd - ad)
    if error > 1e-8 + 1e-3 * abs(ad):
        raise RuntimeError(f"Analytic finite difference failed: {ad=}, {fd=}")
    return {"autograd": ad, "finite_difference": fd, "absolute_error": error}


@torch.no_grad()
def sample_cloud(codes, channel, samples, batch_size, seed):
    torch.manual_seed(seed)
    m, n = codes.shape
    x = codes[:, None, :].expand(m, samples, n).reshape(-1, n)
    return torch.cat([channel(chunk) for chunk in x.split(batch_size)]).reshape(m, samples, n)


def checkpoint_paths(channel, seed):
    lower = channel.lower()
    old = ROOT / "results/journal_wflow_paper_hpc_20260519_064227"
    fixed = ROOT / "results/journal_wflow_fiber_fixed_paper_hpc_20260601_161729"
    short = ROOT / "results/journal_wflow_sspa_budget_screen_20260602_071106"
    baseline = ROOT / "results/journal_baseline_implants_20260602_131336"
    def drift(suite, method):
        return suite / method / f"seed{seed}/checkpoints/enhanced_direct_{lower}_seed{seed}.pt"
    paths = {"kernel_target": drift(old, "kernel_target"),
             "joint_sinkhorn": drift(old, "joint_sinkhorn"),
             "fiber_sinkhorn": drift(short if channel == "SSPA" else fixed, "fiber_sinkhorn"),
             "wgan": baseline / f"wgan/seed{seed}/checkpoints/wgan_{lower}_seed{seed}.pt"}
    for steps in (10, 100):
        paths[f"ddim{steps}"] = baseline / f"diffusion/seed{seed}/checkpoints/diffusion_{lower}_seed{seed}.pt"
    if channel == "SSPA":
        paths["fiber_full"] = drift(fixed, "fiber_sinkhorn")
    return paths


def file_record(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_report(out, report):
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Local gradient-fidelity pilot", "",
        "Existing checkpoint seed only. Independent Monte Carlo repeats are not training seeds.",
        "All methods use the same frozen codec within each comparison. Gradients are expected",
        "cross-entropy gradients, not samplewise Jacobian comparisons between different latent draws.",
        "Full-alphabet centering and unit RMS replace minibatch power normalization for this test.",
        "Noise stays fixed under perturbations. These are not reruns of the paper's SER protocol.", ""]
    for key, result in report["comparisons"].items():
        lines += [f"## {key}", "", f"Analytic-channel baseline CE: {result['baseline_ce']['mean']:.6g}; "
                  f"SER: {result['baseline_ser']['mean']:.6g}.", "",
            "| Model | Conditional SWD | Global SWD at codewords | Input gradient cosine | Encoder gradient cosine | Encoder norm ratio | Relative input-gradient error |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in result["methods"].items():
            lines.append(f"| {name} | {row['conditional_swd']:.5g} | {row['global_swd']:.5g} | "
                         f"{row['input_alignment']['cosine']:.4f} | {row['encoder_alignment']['cosine']:.4f} | "
                         f"{row['encoder_alignment']['norm_ratio']:.3g} | "
                         f"{row['input_alignment']['relative_error']:.3g} |")
        lines += ["", "Analytic row is an independent, equal-sample Monte Carlo gradient and SWD floor.",
                  "Reference gradients use more samples; uncertainty and repeat-level values are in JSON.", "",
                  "### Single Encoder Steps on the Analytic Channel", "",
                  "Decoder frozen; normalized parameter steps, followed by full-codebook normalization.",
                  "Step normalization isolates direction and does not test the effect of erroneous gradient magnitudes.",
                  "Paired common-noise evaluation is independent of gradient estimation.",
                  "CE delta uncertainty is Monte Carlo SE across repeats, not training-seed variability.", "",
                  "| Model | Relative parameter step | Analytic CE delta | MC SE | First-order prediction | SER after step |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, row in result["methods"].items():
            for step in row["steps"]:
                lines.append(f"| {name} | {step['fraction']:.2g} | {step['ce_delta']['mean']:.5g} | "
                             f"{step['ce_delta']['mc_se']:.2g} | {step['predicted_ce_delta']:.5g} | "
                             f"{step['ser']['mean']:.5g} |")
        lines += [""]
    (out / "README.md").write_text("\n".join(lines))


def run_comparison(args, channel, codec_source, out):
    device = torch.device(args.device)
    ae_path = ROOT / f"results/journal_wflow_curves_20260602_220608/{channel}/seed{args.seed}/{codec_source}/symbolic_autoencoder.pt"
    encoder, decoder, payload = load_symbolic_checkpoint(ae_path, device)
    encoder.eval()
    decoder.eval().requires_grad_(False)
    cfg = payload["config"]
    m = int(cfg["message_dim"])
    messages = torch.eye(m, device=device)
    codes = normalized_codebook(encoder, messages).detach()
    summary = payload["summary"]
    noise_std = math.sqrt(1 / (2 * float(summary["rate"]) * 10**(float(summary["ebno_db"]) / 10)))
    analytic = physical_channel(channel, noise_std)
    result = {"codec": file_record(ae_path), "codec_config": cfg, "noise_std": noise_std,
              "ebno_db": summary["ebno_db"], "rate": summary["rate"],
              "finite_difference": finite_difference_check(codes, decoder, analytic), "methods": {}}
    references, baselines = [], []
    for rep in range(args.repeats):
        g, _ = expected_loss(codes, decoder, analytic, args.reference_samples, args.batch_size, 10000 + rep)
        references.append(g)
        _, b = expected_loss(codes, decoder, analytic, args.eval_samples, args.batch_size, 20000 + rep, False)
        baselines.append(b)
    reference = torch.stack(references).mean(0)
    parameter_reference = encoder_gradient(encoder, messages, reference)
    result["reference_split_alignment"] = alignment(torch.stack(references[::2]).mean(0), torch.stack(references[1::2]).mean(0))
    result["reference_encoder_split_alignment"] = alignment(
        encoder_gradient(encoder, messages, torch.stack(references[::2]).mean(0)),
        encoder_gradient(encoder, messages, torch.stack(references[1::2]).mean(0)))
    result["baseline_ce"] = mean_se([b["ce"] for b in baselines])
    result["baseline_ser"] = mean_se([b["ser"] for b in baselines])
    result["baseline_repeats"] = baselines
    true_cloud = sample_cloud(codes, analytic, args.metric_samples, args.batch_size, 30000)
    saved_params = [p.detach().clone() for p in encoder.parameters()]
    parameter_norm = torch.cat([p.flatten() for p in saved_params]).norm()
    tensors = {"codebook": codes.cpu(), "reference_gradients": torch.stack(references).cpu(),
               "parameter_reference": parameter_reference.cpu()}
    paths = checkpoint_paths(channel, args.seed)
    names = args.methods.split(",")
    if channel == "SSPA" and "fiber_sinkhorn" in names and "fiber_full" not in names:
        names.append("fiber_full")
    for name in names:
        print(f"[{channel}/{codec_source}] {name}", flush=True)
        start = time.monotonic()
        implant_cfg = None
        if name != "analytic":
            checkpoint = torch.load(paths[name], map_location="cpu", weights_only=False)
            implant_cfg = checkpoint.get("config", {}) | checkpoint.get("metadata", {}).get("config", {})
            if int(implant_cfg["n"]) != codes.shape[1] or not math.isclose(float(implant_cfg["noise_std"]), noise_std, rel_tol=1e-6):
                raise ValueError(f"Incompatible channel dimension/noise calibration: {paths[name]}")
            del checkpoint
        implant = analytic if name == "analytic" else load_implant_from_checkpoint(
            paths[name], device=device, diffusion_sampler="ddim",
            ddim_steps=int(name[4:]) if name.startswith("ddim") else 100)
        gradients, stats = [], []
        for rep in range(args.repeats):
            g, s = expected_loss(codes, decoder, implant, args.gradient_samples, args.batch_size, 40000 + rep)
            gradients.append(g)
            stats.append(s)
        gradient = torch.stack(gradients).mean(0)
        pg = encoder_gradient(encoder, messages, gradient)
        cloud = sample_cloud(codes, implant, args.metric_samples, args.batch_size, 50000)
        row = {"checkpoint": None if name == "analytic" else file_record(paths[name]),
               "implant_config": implant_cfg,
               "input_alignment": alignment(gradient, reference), "encoder_alignment": alignment(pg, parameter_reference),
               "repeat_alignment": [alignment(g, reference) for g in gradients], "sample_repeats": stats,
               "conditional_swd": conditional_anchor_swd(true_cloud, cloud, num_projections=args.projections, seed=60000),
               "global_swd": sliced_wasserstein_distance(true_cloud.flatten(0, 1), cloud.flatten(0, 1), num_projections=args.projections, seed=60000),
               "steps": []}
        tensors[name] = {"gradients": torch.stack(gradients).cpu(), "parameter_gradient": pg.cpu()}
        direction = pg / pg.norm().clamp_min(1e-30)
        try:
            for fraction in args.steps:
                size = fraction * parameter_norm
                offset = 0
                with torch.no_grad():
                    for p, original in zip(encoder.parameters(), saved_params):
                        p.copy_(original - size * direction[offset:offset + p.numel()].reshape_as(p))
                        offset += p.numel()
                    new_codes = normalized_codebook(encoder, messages)
                evaluations = [expected_loss(new_codes, decoder, analytic, args.eval_samples, args.batch_size, 20000 + rep, False)[1]
                               for rep in range(args.repeats)]
                row["steps"].append({"fraction": fraction, "parameter_step_norm": size.item(),
                    "codeword_rms_displacement": (new_codes - codes).square().mean().sqrt().item(),
                    "predicted_ce_delta": float(-size * parameter_reference.dot(direction)),
                    "ce_delta": mean_se([v["ce"] - b["ce"] for v, b in zip(evaluations, baselines)]),
                    "ser": mean_se([v["ser"] for v in evaluations]), "evaluation_repeats": evaluations})
        finally:
            with torch.no_grad():
                for p, original in zip(encoder.parameters(), saved_params):
                    p.copy_(original)
        row["seconds"] = time.monotonic() - start
        result["methods"][name] = row
        print(f"  cosine={row['input_alignment']['cosine']:.4f}, encoder={row['encoder_alignment']['cosine']:.4f}, "
              f"anchor SWD={row['conditional_swd']:.5g}, {row['seconds']:.1f}s", flush=True)
        (out / f"{channel}_{codec_source}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        torch.save(tensors, out / f"{channel}_{codec_source}_gradients.pt")
        del implant
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default="AWGN,SSPA")
    parser.add_argument("--codec-sources", default="analytic")
    parser.add_argument("--methods", default="analytic,kernel_target,joint_sinkhorn,fiber_sinkhorn,wgan,ddim10,ddim100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--gradient-samples", type=int, default=2048)
    parser.add_argument("--reference-samples", type=int, default=16384)
    parser.add_argument("--eval-samples", type=int, default=16384)
    parser.add_argument("--metric-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--projections", type=int, default=64)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.0001, 0.0003, 0.001])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.repeats < 2 or min(args.gradient_samples, args.reference_samples, args.eval_samples, args.metric_samples, args.batch_size, args.projections) < 1:
        parser.error("At least two repeats and positive sample/batch/projection counts are required.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"config": vars(args), "torch": torch.__version__,
              "hardware": torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script": file_record(Path(__file__).resolve()), "comparisons": {}}
    for channel in args.channels.split(","):
        for source in args.codec_sources.split(","):
            key = f"{channel}/{source}"
            report["comparisons"][key] = run_comparison(args, channel, source, out)
            write_report(out, report)
    print(out / "README.md", flush=True)


if __name__ == "__main__":
    main()
