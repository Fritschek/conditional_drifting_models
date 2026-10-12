"""Shared evaluation and resumable downstream training for channel comparisons."""

import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .fast_channels import physical_channel, channel_spec
from ..metrics import (sliced_wasserstein_distance, conditional_anchor_swd,
    conditional_anchor_gaussian_w2, conditional_anchor_mean_l2, conditional_anchor_cov_fro)
from ..sspa_trajectory import isolated_rng, rng_state, restore_rng
from ..symbolic_ae import SymbolicEncoder, SymbolicDecoder
from ..training import set_seed
from scripts.run_local_gradient_fidelity import normalized_codebook, expected_loss, alignment, encoder_gradient


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def distribution_metrics(channel, sampler, device, seed, size, anchors, samples, projections, batch=1024, codes=None):
    n = channel_spec(channel)["n"]
    with isolated_rng(seed):
        x = torch.randn(size, n, device=device)
        truth = physical_channel(channel, x)
        truth2 = physical_channel(channel, x)
        inputs = torch.randn(anchors, n, device=device) if codes is None else codes.detach()
        anchors = len(inputs)
        repeated = inputs.repeat_interleave(samples, 0)
        cloud = physical_channel(channel, repeated).reshape(anchors, samples, n)
        floor = physical_channel(channel, repeated).reshape_as(cloud)
        generated = torch.cat([sampler(v) for v in x.split(batch)])
        estimated = torch.cat([sampler(v) for v in repeated.split(batch)]).reshape_as(cloud)
        def compare(a, b):
            return dict(swd=conditional_anchor_swd(a, b, num_projections=projections, seed=seed),
                gaussian_w2=conditional_anchor_gaussian_w2(a, b), mean_l2=conditional_anchor_mean_l2(a, b),
                covariance_fro=conditional_anchor_cov_fro(a, b))
        return dict(global_swd=sliced_wasserstein_distance(truth, generated, num_projections=projections, seed=seed),
            global_floor=sliced_wasserstein_distance(truth, truth2, num_projections=projections, seed=seed),
            conditional=compare(cloud, estimated), analytic_floor=compare(cloud, floor),
            global_samples=size, anchors=anchors, samples_per_anchor=samples, projections=projections)


def timings(sampler, n, device, seed, batch_sizes, repeats):
    rows = []
    with isolated_rng(seed):
        for batch in batch_sizes:
            x = torch.randn(batch, n, device=device)
            for backward in (False, True):
                values = []
                for index in range(repeats + 3):
                    sync(device)
                    start = time.perf_counter()
                    with torch.set_grad_enabled(backward):
                        v = x.detach().requires_grad_(backward)
                        y = sampler(v)
                        if backward:
                            gradient = torch.autograd.grad(y.square().mean(), v)[0]
                            if not torch.isfinite(gradient).all():
                                raise FloatingPointError("Nonfinite implant input gradient")
                    sync(device)
                    if index >= 3:
                        values.append(1000 * (time.perf_counter() - start))
                rows.append(dict(batch=batch, operation="forward_and_input_backward" if backward else "sampling",
                    milliseconds_per_batch=values, mean_ms=float(np.mean(values)),
                    std_ms=float(np.std(values, ddof=1)) if repeats > 1 else None,
                    median_ms=float(np.median(values)), p95_ms=float(np.quantile(values, .95)),
                    amortized_ms_per_sample=float(np.mean(values) / batch)))
    return rows


def codec(channel, device, seed):
    set_seed(seed)
    spec = channel_spec(channel)
    encoder = SymbolicEncoder(spec["message_dim"], spec["n"], 16, normalization="none").to(device)
    decoder = SymbolicDecoder(spec["message_dim"], spec["n"], 16).to(device)
    return encoder, decoder, torch.eye(spec["message_dim"], device=device)


def train_codec(path, channel, sampler, device, seed, updates, batch, save_every, resume):
    encoder, decoder, messages = codec(channel, device, seed)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.NAdam(parameters, lr=.001)
    labels_rng = torch.Generator(device=device).manual_seed(seed + 700000)
    start_update, seconds = 0, 0.
    if Path(path).exists():
        if not resume:
            raise FileExistsError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        labels_rng.set_state(state["labels_rng"].cpu())
        restore_rng(state["rng"])
        start_update, seconds = state["update"], state["seconds"]
    for update in range(start_update, updates):
        sync(device)
        start = time.perf_counter()
        labels = torch.randint(len(messages), (batch,), device=device, generator=labels_rng)
        codes = normalized_codebook(encoder, messages)
        loss = F.cross_entropy(decoder(sampler(codes[labels])), labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        seconds += time.perf_counter() - start
        if (update + 1) % save_every == 0 or update + 1 == updates:
            save(path, dict(encoder=encoder.state_dict(), decoder=decoder.state_dict(),
                optimizer=optimizer.state_dict(), labels_rng=labels_rng.get_state(), rng=rng_state(),
                update=update + 1, seconds=seconds, channel=channel, seed=seed))
    return encoder, decoder, messages, seconds


def evaluate_codec(channel, encoder, decoder, messages, samples, batch, seed, offsets):
    codes = normalized_codebook(encoder, messages).detach()
    nominal = channel_spec(channel)["ebno_db"]
    rows = []
    with isolated_rng(seed):
        for offset in offsets:
            ebno = nominal + offset
            _, result = expected_loss(codes, decoder, lambda x: physical_channel(channel, x, ebno),
                samples, batch, seed, gradient=False)
            rows.append(dict(ebno_db=ebno, **result))
    return rows


def common_codec_gradients(channel, sampler, encoder, decoder, messages, samples, batch, seed):
    """Compare at one shared analytic-trained codec, including independent MC references."""
    codes = normalized_codebook(encoder, messages).detach()
    with isolated_rng(seed):
        g1, _ = expected_loss(codes, decoder, lambda x: physical_channel(channel, x), samples, batch, seed)
        g2, _ = expected_loss(codes, decoder, lambda x: physical_channel(channel, x), samples, batch, seed + 1)
        g, loss = expected_loss(codes, decoder, sampler, samples, batch, seed + 2)
        e1, e2, e = [encoder_gradient(encoder, messages, v) for v in (g1, g2, g)]
    return dict(code_gradient=alignment(g, g1), encoder_gradient=alignment(e, e1),
        analytic_repeat_code=alignment(g2, g1), analytic_repeat_encoder=alignment(e2, e1),
        surrogate_loss=loss, samples_per_message=samples,
        interpretation="Finite-Monte-Carlo comparisons, not a resolved population gradient claim")
