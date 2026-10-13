"""Shared validation and paired evaluation for the October coverage follow-up."""

import itertools
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from conditional_drifting.baselines.fast_channels import FastImplant, channel_spec
from conditional_drifting.baselines.fast_evaluation import write_json, train_codec
from conditional_drifting.baselines.official_sources import load_sources, sha256
from conditional_drifting.paper2309_presets import ebno_to_noise
from conditional_drifting.channels import sspa
from conditional_drifting.sspa_trajectory import isolated_rng
from conditional_drifting.sspa_update_screen import load_official, OFFICIAL_COMMIT
from scripts.run_fast_channel_comparison import train
from scripts.run_local_gradient_fidelity import normalized_codebook
from scripts.run_sspa_input_coverage import CoverageTrainer, load_codec

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("analytic", "gaussian_normal", "gaussian_broad", "sinkhorn_normal", "sinkhorn_broad")


def read(path):
    return json.loads(Path(path).read_text())


def validate_parent(parent, source_dir, wflow_source):
    manifest = read(parent / "manifest.json")
    if manifest["settings"]["channel"] != "SSPA":
        raise ValueError("Expected the completed SSPA application parent")
    for name, digest in manifest["adapter_files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen parent source changed: {name}")
    official, wflow = load_sources(source_dir), load_official(wflow_source)
    if official.records != manifest["sources"] or manifest["wflow"] != dict(
            commit=OFFICIAL_COMMIT, sha256=sha256(wflow_source / "drift_loss_ot.py")):
        raise ValueError("Upstream source differs from parent")
    return manifest, official, wflow


def runtime(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit CPU fallback")
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, threads=torch.get_num_threads(),
                device=device, hardware=torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu")


def file_hashes(paths):
    return {str(Path(p).resolve().relative_to(ROOT)): sha256(p) for p in paths}


def cached(path, stamp, resume):
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(path)
    value = read(path)
    if value["stamp"] != stamp:
        raise ValueError(f"Checkpoint or manifest changed for {path}")
    return value["result"]


def record(path, stamp, result):
    write_json(path, dict(stamp=stamp, result=result))


class MeasuredCoverageTrainer(CoverageTrainer):
    """Keep the frozen updates, additionally count characterization-input energy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_energy_sum = 0.
        self.input_norm_max = 0.

    def inputs(self, anchors):
        x = super().inputs(anchors)
        norm2 = x.detach().double().square().sum(1)
        self.input_energy_sum += norm2.sum().item()
        self.input_norm_max = max(self.input_norm_max, norm2.max().sqrt().item())
        return x

    def state(self):
        return dict(super().state(), input_energy_sum=self.input_energy_sum, input_norm_max=self.input_norm_max)

    def restore(self, state):
        super().restore(state)
        self.input_energy_sum, self.input_norm_max = state["input_energy_sum"], state["input_norm_max"]


def finish_generator(trainer, path, save_every, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)["trainer"]
        trainer.restore(state)
        if trainer.update > trainer.cfg.updates:
            raise ValueError("Checkpoint exceeds fixed budget")
        if trainer.update == trainer.cfg.updates:
            return
    train(trainer, path, save_every, resume)


def finish_codec(path, sampler, device, seed, settings, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["channel"] != "SSPA" or state["seed"] != seed or state["update"] > settings["ae_updates"]:
            raise ValueError("Codec checkpoint identity/budget mismatch")
        if state["update"] == settings["ae_updates"]:
            return load_codec(path, device)
    return train_codec(path, "SSPA", sampler, device, seed, settings["ae_updates"],
                       settings["ae_batch"], settings["save_every"], resume)


@torch.no_grad()
def paired_curves(codecs, samples, batch, seed, ebnos=(4., 6., 8.)):
    """Balanced labels and identical standard noise across codecs and SNRs."""
    if samples < 1 or batch < 64 or not codecs:
        raise ValueError("Positive per-message trials and batch >= 64 required")
    codes = {name: normalized_codebook(e, messages).detach() for name, (e, d, messages, _) in codecs.items()}
    first = next(iter(codes.values()))
    m, n = first.shape
    if (m, n) != (64, 8) or any(x.shape != first.shape for x in codes.values()):
        raise ValueError("Expected M64/n8 codebooks")
    result = []
    for ebno in ebnos:
        stats = {name: dict(errors=0, bit_errors=0, ce_sum=0., errors_per_message=[0]*m,
                           bit_errors_per_message=[0]*m) for name in codecs}
        pairs = {f"{a}:{b}": dict(a=a, b=b, a_only_wrong=0, b_only_wrong=0, both_wrong=0,
                                 neither_wrong=0) for a, b in itertools.combinations(codecs, 2)}
        sigma = ebno_to_noise(ebno, channel_spec("SSPA")["rate"]) / math.sqrt(2)
        with isolated_rng(seed):
            rng = torch.Generator(device=first.device).manual_seed(seed)
            # Zero-noise sspa still draws RNG internally; the explicit stream stays independent.
            means = {name: sspa(x, 0., first.device) for name, x in codes.items()}
            for start in range(0, samples, batch // m):
                count = min(batch // m, samples-start)
                labels = torch.arange(m, device=first.device).repeat_interleave(count)
                noise = torch.randn(m*count, n, generator=rng, device=first.device)
                errors = {}
                for name, (_, decoder, _, _) in codecs.items():
                    logits = decoder(means[name].repeat_interleave(count, 0) + sigma*noise)
                    pred = logits.argmax(-1)
                    wrong = pred != labels
                    bits = sum(((pred ^ labels) >> bit) & 1 for bit in range(6))
                    row = stats[name]
                    row["errors"] += wrong.sum().item()
                    row["bit_errors"] += bits.sum().item()
                    row["ce_sum"] += F.cross_entropy(logits, labels, reduction="sum").double().item()
                    for key, values in (("errors_per_message", wrong), ("bit_errors_per_message", bits)):
                        row[key] = [a+b for a, b in zip(row[key], values.reshape(m, count).sum(1).tolist())]
                    errors[name] = wrong
                for pair in pairs.values():
                    a, b = errors[pair["a"]], errors[pair["b"]]
                    pair["a_only_wrong"] += (a & ~b).sum().item()
                    pair["b_only_wrong"] += (~a & b).sum().item()
                    pair["both_wrong"] += (a & b).sum().item()
                    pair["neither_wrong"] += (~a & ~b).sum().item()
        trials = m*samples
        for row in stats.values():
            row.update(trials=trials, ser=row["errors"]/trials, ber=row["bit_errors"]/(6*trials),
                       ce=row.pop("ce_sum")/trials)
        for pair in pairs.values():
            delta = (pair["a_only_wrong"]-pair["b_only_wrong"])/trials
            second = (pair["a_only_wrong"]+pair["b_only_wrong"])/trials
            pair.update(trials=trials, ser_a_minus_b=delta,
                        pooled_paired_mc_se=math.sqrt(max(0., second-delta*delta)/(trials-1)))
        result.append(dict(ebno_db=ebno, variants=stats, pairs=list(pairs.values())))
    return result


class CommonNoiseImplant(FastImplant):
    """Separate initial-latent stream from DDIM's internal eta=0 RNG consumption."""

    def reset(self, seed):
        self.noise_rng = torch.Generator(device=self.trainer.device).manual_seed(seed)
        return self

    def forward(self, x):
        noise = torch.randn(len(x), self.trainer.latent_dim, device=x.device, dtype=x.dtype,
                            generator=self.noise_rng)
        return self.trainer.sample(x, self.steps, noise=noise, model=self.model)
