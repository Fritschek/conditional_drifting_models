"""Paired, resumable SSPA codec trials and frozen-codec gradient checks."""

import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .channels import sspa
from .model import ConditionalDriftingGenerator
from .symbolic_ae import SymbolicEncoder, SymbolicDecoder
from .sspa_trajectory import isolated_rng, rng_state, restore_rng
from .training import set_seed
from .metrics import conditional_anchor_swd, conditional_anchor_gaussian_w2
from scripts.run_local_gradient_fidelity import (
    normalized_codebook, expected_loss, encoder_gradient, alignment,
    sample_cloud, finite_difference_check)

NOISE_STD = .3250531435997416


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def analytic_channel(x, ebno=8.):
    return sspa(x, NOISE_STD * 10 ** ((8. - ebno) / 20.), x.device)


def load_generator(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg["n"] != 8 or not math.isclose(cfg["noise_std"], NOISE_STD, rel_tol=1e-9):
        raise ValueError("SSPA generator dimension/noise mismatch")
    model = ConditionalDriftingGenerator(8, 8, cfg["latent_dim"], cfg["hidden_dim"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


class CodecTrial:
    def __init__(self, seed, channel, device, contract, batch_size=500):
        self.seed, self.channel, self.device, self.contract = seed, channel, torch.device(device), contract
        self.batch_size = batch_size
        set_seed(seed)
        self.encoder = SymbolicEncoder(64, 8, 16, normalization="none").to(device)
        self.decoder = SymbolicDecoder(64, 8, 16).to(device)
        self.optimizer = torch.optim.NAdam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=.001)
        self.messages = torch.eye(64, device=device)
        self.labels = torch.Generator(device=device).manual_seed(1000000 + seed)
        torch.manual_seed(2000000 + seed)
        self.update = 0
        self.history, self.trace = [], []
        self.train_seconds, self.eval_seconds = 0., 0.

    def step(self, trace=False):
        labels = torch.randint(64, (self.batch_size,), device=self.device, generator=self.labels)
        codes = normalized_codebook(self.encoder, self.messages)
        loss = F.cross_entropy(self.decoder(self.channel(codes[labels])), labels)
        self.optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                              1., error_if_nonfinite=True)
        self.optimizer.step()
        self.update += 1
        if trace:
            self.trace.append(dict(update=self.update, loss=loss.item(), gradient_norm=norm.item()))

    def evaluate(self, samples, seed, ebno=8.):
        with isolated_rng(seed), torch.no_grad():
            codes = normalized_codebook(self.encoder, self.messages)
            _, stats = expected_loss(codes, self.decoder, lambda x: analytic_channel(x, ebno),
                                     samples, 4096, seed, False)
        return stats | {"ebno_db": ebno, "bit_trials": 6 * stats["trials"]}

    def save(self, path):
        atomic_save(dict(contract=self.contract, seed=self.seed, batch_size=self.batch_size,
            encoder=self.encoder.state_dict(), decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(), update=self.update, rng=rng_state(),
            labels_rng=self.labels.get_state(), history=self.history, trace=self.trace,
            train_seconds=self.train_seconds, eval_seconds=self.eval_seconds), path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["contract"] != self.contract or state["seed"] != self.seed or state["batch_size"] != self.batch_size:
            raise ValueError("Codec resume contract mismatch")
        for name in ("encoder", "decoder", "optimizer"):
            getattr(self, name).load_state_dict(state[name])
        for name in ("update", "history", "trace", "train_seconds", "eval_seconds"):
            setattr(self, name, state[name])
        self.labels.set_state(state["labels_rng"].cpu())
        restore_rng(state["rng"])


def load_codec(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    encoder = SymbolicEncoder(64, 8, 16, normalization="none").to(device)
    decoder = SymbolicDecoder(64, 8, 16).to(device)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    return encoder.eval(), decoder.eval().requires_grad_(False)


def gradient_comparison(codec_path, generators, device, smoke=False, budget_check=lambda: None):
    encoder, decoder = load_codec(codec_path, device)
    messages = torch.eye(64, device=device)
    codes = normalized_codebook(encoder, messages).detach()
    repeats, samples, reference_samples, metric_samples = (2, 8, 16, 16) if smoke else (4, 8192, 65536, 4096)
    result = dict(codec=str(codec_path), repeats=repeats, gradient_samples_per_message=samples,
        reference_samples_per_message=reference_samples, metric_samples_per_message=metric_samples,
        finite_difference=finite_difference_check(codes, decoder, analytic_channel), methods={})
    references = []
    for rep in range(repeats):
        budget_check()
        references.append(expected_loss(codes, decoder, analytic_channel, reference_samples, 4096, 710000 + rep)[0])
    reference = torch.stack(references).mean(0)
    parameter_references = torch.stack([encoder_gradient(encoder, messages, g) for g in references])
    parameter_reference = parameter_references.mean(0)
    # MC standard error of the mean vector, not an independent training-seed CI.
    input_se = torch.stack(references).var(0, unbiased=True).sum().sqrt().item() / math.sqrt(repeats)
    parameter_se = parameter_references.var(0, unbiased=True).sum().sqrt().item() / math.sqrt(repeats)
    input_resolved = reference.norm().item() > 5 * input_se
    parameter_resolved = parameter_reference.norm().item() > 5 * parameter_se
    result.update(reference_input_norm=reference.norm().item(), reference_input_mc_se_norm=input_se,
        reference_encoder_norm=parameter_reference.norm().item(), reference_encoder_mc_se_norm=parameter_se,
        reference_input_resolved=input_resolved, reference_encoder_resolved=parameter_resolved,
        reference_split_alignment=alignment(torch.stack(references[::2]).mean(0), torch.stack(references[1::2]).mean(0)))
    true = sample_cloud(codes, analytic_channel, metric_samples, 4096, 720000)
    other = sample_cloud(codes, analytic_channel, metric_samples, 4096, 720001)
    projections = 4 if smoke else 64
    result["analytic_floors"] = dict(swd=conditional_anchor_swd(true, other, num_projections=projections, seed=730000),
        gw2=conditional_anchor_gaussian_w2(true, other))
    tensors = dict(codes=codes.cpu(), references=torch.stack(references).cpu(),
                   encoder_references=parameter_references.cpu(), methods={})
    for name, path in {"analytic": None, **generators}.items():
        budget_check()
        channel = analytic_channel if path is None else load_generator(path, device)
        start = time.perf_counter()
        grads, stats = [], []
        for rep in range(repeats):
            budget_check()
            gradient, values = expected_loss(codes, decoder, channel, samples, 4096, 740000 + rep)
            grads.append(gradient)
            stats.append(values)
        gradient = torch.stack(grads).mean(0)
        pg = encoder_gradient(encoder, messages, gradient)
        cloud = sample_cloud(codes, channel, metric_samples, 4096, 750000)
        input_metrics, parameter_metrics = alignment(gradient, reference), alignment(pg, parameter_reference)
        # Retain absolute errors even when the reference is too small for ratios.
        if not input_resolved:
            input_metrics = {key: None for key in input_metrics}
        if not parameter_resolved:
            parameter_metrics = {key: None for key in parameter_metrics}
        result["methods"][name] = dict(checkpoint=None if path is None else str(path),
            input_alignment=input_metrics, encoder_alignment=parameter_metrics,
            input_absolute_error=(gradient - reference).norm().item(),
            encoder_absolute_error=(pg - parameter_reference).norm().item(),
            input_repeat_noise_norm=torch.stack(grads).var(0, unbiased=True).sum().sqrt().item(),
            conditional_swd=conditional_anchor_swd(true, cloud, num_projections=projections, seed=730000),
            conditional_gw2=conditional_anchor_gaussian_w2(true, cloud),
            sample_repeats=stats, seconds=time.perf_counter() - start)
        tensors["methods"][name] = dict(gradients=torch.stack(grads).cpu(), encoder_gradient=pg.cpu())
    return result, tensors
