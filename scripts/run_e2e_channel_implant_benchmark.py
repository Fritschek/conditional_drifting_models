from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditional_drifting.e2e_implants import AnalyticAWGNImplant, AnalyticChannelImplant, load_implant_from_checkpoint
from conditional_drifting.weight_registry import build_artifact_path, register_artifact, resolve_registered_artifact

VENDORED_TURBO_ROOT = ROOT / "external" / "turbo_mingru_decoder"
DEFAULT_TURBO_ROOT = VENDORED_TURBO_ROOT if VENDORED_TURBO_ROOT.exists() else ROOT.parent / "turbo_mingru_decoder"


def add_turbo_repo_to_path(repo_root: Path) -> None:
    resolved = str(repo_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def generate_data(batch_size: int, sequence_length: int, num_symbols: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, num_symbols, (batch_size, sequence_length), dtype=torch.float32, device=device)


def compute_ber(decoded_output: torch.Tensor, inputs: torch.Tensor) -> float:
    binary_predictions = torch.round(decoded_output)
    prediction_errors = torch.ne(binary_predictions, inputs)
    return torch.mean(prediction_errors.float()).detach().cpu().item()


def compute_bler(decoded_output: torch.Tensor, inputs: torch.Tensor) -> float:
    binary_predictions = torch.round(decoded_output)
    block_errors = torch.ne(binary_predictions, inputs).any(dim=1)
    return torch.mean(block_errors.float()).detach().cpu().item()


def build_autoencoder(
    repo_root: Path,
    model_type: str,
    batch_size: int,
    sequence_length: int,
    enc_num_layers: int,
    dec_num_layers: int,
    hidden_size_gru: int,
    num_iteration: int,
    num_iter_ft_cnn: int,
    device: torch.device,
):
    add_turbo_repo_to_path(repo_root)
    model_tae = importlib.import_module("model_turboAE")

    config_params = model_tae.TurboConfig(
        block_len=sequence_length,
        enc_num_unit=100,
        dec_num_unit=100,
        batch_size=batch_size,
        enc_num_layer=enc_num_layers,
        dec_num_layer=dec_num_layers,
        hidden_size_gru=hidden_size_gru,
        num_iteration=num_iteration,
        num_iter_ft_cnn=num_iter_ft_cnn,
    )
    interleaver = model_tae.Interleaver(config_params).to(device)

    if model_type == "cnn_turbo":
        encoder = model_tae.ENC_CNNTurbo(config_params, interleaver).to(device)
        decoder = model_tae.DEC_CNNTurbo(config_params, interleaver).to(device)
    elif model_type == "CNN_turbo_serial":
        encoder = model_tae.ENC_CNNTurbo_serial(config_params, interleaver).to(device)
        decoder = model_tae.DEC_CNNTurbo_serial(config_params, interleaver).to(device)
    elif model_type == "gru_turbo":
        encoder = model_tae.ENC_GRUTurbo(config_params, interleaver).to(device)
        decoder = model_tae.DEC_CNNTurbo(config_params, interleaver).to(device)
    elif model_type == "gru_turbo_full":
        encoder = model_tae.ENC_GRUTurbo(config_params, interleaver).to(device)
        decoder = model_tae.DEC_CNNTurboXminGRU(config_params, interleaver).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return encoder, decoder


def build_implant_from_spec(
    *,
    mode: str,
    checkpoint: str | None,
    name: str | None,
    weights_root: str,
    device: torch.device,
    diffusion_sampler: str,
    ddim_steps: int,
    channel: str,
):
    if mode == "analytic_awgn":
        return AnalyticAWGNImplant()
    if mode == "analytic_channel":
        return AnalyticChannelImplant(channel)
    if mode == "checkpoint":
        implant_checkpoint = checkpoint
        if name:
            implant_checkpoint = resolve_registered_artifact(name, weights_root)["checkpoint"]
        if not implant_checkpoint:
            raise ValueError("Provide a checkpoint path or registered artifact name for checkpoint implants.")
        return load_implant_from_checkpoint(
            implant_checkpoint,
            device=device,
            diffusion_sampler=diffusion_sampler,
            ddim_steps=ddim_steps,
        )
    raise ValueError(f"Unsupported implant mode: {mode}")


def apply_implant(
    implant,
    encoded_data: torch.Tensor,
    *,
    ebno_db: float,
    rate: float,
    device: torch.device,
    decoder_training: bool = False,
    ebno_range: tuple[float, float] = (-3.5, 0.0),
) -> torch.Tensor:
    return implant(
        encoded_data,
        ebno_db=ebno_db,
        rate=rate,
        device=device,
        decoder_training=decoder_training,
        ebno_range=ebno_range,
    )


def evaluate_model(
    encoder,
    decoder,
    implant,
    *,
    eval_batches: int,
    batch_size: int,
    sequence_length: int,
    num_symbols: int,
    ebno_db: float,
    rate: float,
    device: torch.device,
    amp: bool = False,
    amp_dtype: str = "bfloat16",
) -> dict:
    encoder.eval()
    decoder.eval()
    losses = []
    ber_values = []
    bler_values = []
    with torch.no_grad():
        for _ in range(eval_batches):
            input_data = generate_data(batch_size, sequence_length, num_symbols, device)
            with make_autocast_context(device, enabled=amp, amp_dtype=amp_dtype):
                encoded_data = encoder(input_data)
                noisy_data = apply_implant(implant, encoded_data, ebno_db=ebno_db, rate=rate, device=device)
                logits = decoder(noisy_data)
            probs = torch.sigmoid(logits)
            losses.append(float(F.binary_cross_entropy_with_logits(logits, input_data).item()))
            ber_values.append(compute_ber(probs, input_data))
            bler_values.append(compute_bler(probs, input_data))
    return {"loss": float(np.mean(losses)), "ber": float(np.mean(ber_values)), "bler": float(np.mean(bler_values))}


def make_autocast_context(device: torch.device, *, enabled: bool, amp_dtype: str):
    if not enabled or not str(device).startswith("cuda"):
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, enabled=True, dtype=dtype)


def resolve_effective_batches(args: argparse.Namespace) -> dict[str, int]:
    enc_effective_bs = int(args.batch_size)
    dec_effective_bs = int(args.batch_size) * int(args.dec_bs_fac)
    enc_micro_bs = int(args.enc_micro_batch_size) if int(args.enc_micro_batch_size) > 0 else enc_effective_bs
    dec_micro_bs = int(args.dec_micro_batch_size) if int(args.dec_micro_batch_size) > 0 else dec_effective_bs
    if enc_effective_bs % enc_micro_bs != 0:
        raise ValueError(f"batch_size={enc_effective_bs} must be divisible by enc_micro_batch_size={enc_micro_bs}")
    if dec_effective_bs % dec_micro_bs != 0:
        raise ValueError(f"decoder effective batch={dec_effective_bs} must be divisible by dec_micro_batch_size={dec_micro_bs}")
    return {
        "enc_effective_batch_size": enc_effective_bs,
        "dec_effective_batch_size": dec_effective_bs,
        "enc_micro_batch_size": enc_micro_bs,
        "dec_micro_batch_size": dec_micro_bs,
        "enc_accum_steps": enc_effective_bs // enc_micro_bs,
        "dec_accum_steps": dec_effective_bs // dec_micro_bs,
    }


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_e2e_checkpoint(path: Path, encoder, decoder, args: argparse.Namespace, epoch: int, metrics: dict) -> None:
    payload = {
        "epoch": int(epoch),
        "model_type": args.model_type,
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "config": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "channel_length": args.channel_length,
            "enc_num_layers": args.enc_num_layers,
            "dec_num_layers": args.dec_num_layers,
            "hidden_size_gru": args.hidden_size_gru,
            "num_iteration": args.num_iteration,
            "num_iter_ft_cnn": args.num_iter_ft_cnn,
        },
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def maybe_gpu_warmup(device: torch.device, *, enabled: bool, seconds: float, size: int) -> None:
    if not enabled or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    with torch.no_grad():
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.time()
        while time.time() - t0 < seconds:
            a @ b
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TurboAE model through a pluggable channel implant.")
    parser.add_argument("--turbo-root", type=str, default=str(DEFAULT_TURBO_ROOT))
    parser.add_argument("--model-type", type=str, default="cnn_turbo", choices=["cnn_turbo", "CNN_turbo_serial", "gru_turbo", "gru_turbo_full"])
    parser.add_argument("--channel", type=str, default="AWGN", choices=["AWGN", "Rayleigh", "ModeFlip", "SSPA", "TDL", "OptFib"])
    parser.add_argument("--implant", type=str, default="analytic_awgn", choices=["analytic_awgn", "analytic_channel", "checkpoint"], help="Backward-compatible alias for --train-implant.")
    parser.add_argument("--implant-checkpoint", type=str, default=None, help="Backward-compatible alias for --train-implant-checkpoint.")
    parser.add_argument("--implant-name", type=str, default=None, help="Backward-compatible alias for --train-implant-name.")
    parser.add_argument("--train-implant", type=str, default=None, choices=["analytic_awgn", "analytic_channel", "checkpoint"])
    parser.add_argument("--train-implant-checkpoint", type=str, default=None)
    parser.add_argument("--train-implant-name", type=str, default=None)
    parser.add_argument("--eval-implant", type=str, default=None, choices=["analytic_awgn", "analytic_channel", "checkpoint"])
    parser.add_argument("--eval-implant-checkpoint", type=str, default=None)
    parser.add_argument("--eval-implant-name", type=str, default=None)
    parser.add_argument("--diffusion-sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dec-bs-fac", type=int, default=4)
    parser.add_argument("--enc-micro-batch-size", type=int, default=0)
    parser.add_argument("--dec-micro-batch-size", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--eval-num-blocks", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--channel-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-iterations", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--compile-models", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--enable-gpu-warmup", action="store_true")
    parser.add_argument("--gpu-warmup-seconds", type=float, default=1.5)
    parser.add_argument("--gpu-warmup-size", type=int, default=2048)
    parser.add_argument(
        "--training-regime",
        type=str,
        default="overfit_fresh_noise",
        choices=["overfit_fresh_noise", "epoch_joint", "epoch_alternate", "overnight_alternate"],
    )
    parser.add_argument("--training-mode", type=str, default="alternate", choices=["joint", "alternate"], help=argparse.SUPPRESS)
    parser.add_argument("--ebno-db", type=float, default=4.0)
    parser.add_argument("--decoder-ebno-offset-low", type=float, default=-3.5)
    parser.add_argument("--decoder-ebno-offset-high", type=float, default=0.0)
    parser.add_argument("--enc-num-layers", type=int, default=2)
    parser.add_argument("--dec-num-layers", type=int, default=5)
    parser.add_argument("--hidden-size-gru", type=int, default=4)
    parser.add_argument("--num-iteration", type=int, default=5)
    parser.add_argument("--num-iter-ft-cnn", type=int, default=10)
    parser.add_argument("--num-symbols", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--lr-plateau-patience-evals", type=int, default=0)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-max-reductions", type=int, default=0)
    parser.add_argument("--lr-plateau-min-lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="results/e2e_channel_implant")
    parser.add_argument("--weights-root", type=str, default=str(ROOT / "weights"))
    parser.add_argument("--load-e2e-checkpoint", type=str, default=None)
    parser.add_argument("--load-e2e-name", type=str, default=None)
    parser.add_argument("--save-e2e-name", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if str(device).startswith("cuda"):
        if args.allow_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder = build_autoencoder(
        Path(args.turbo_root),
        args.model_type,
        args.batch_size,
        args.sequence_length,
        args.enc_num_layers,
        args.dec_num_layers,
        args.hidden_size_gru,
        args.num_iteration,
        args.num_iter_ft_cnn,
        device,
    )
    if args.load_e2e_name:
        e2e_checkpoint = resolve_registered_artifact(args.load_e2e_name, args.weights_root)["checkpoint"]
    else:
        e2e_checkpoint = args.load_e2e_checkpoint
    if e2e_checkpoint:
        payload = torch.load(e2e_checkpoint, map_location=device)
        encoder.load_state_dict(payload["encoder_state"])
        decoder.load_state_dict(payload["decoder_state"])
    if args.compile_models and hasattr(torch, "compile"):
        encoder = torch.compile(encoder, mode=args.compile_mode)
        decoder = torch.compile(decoder, mode=args.compile_mode)
    train_implant_mode = args.train_implant or args.implant
    train_implant_checkpoint = args.train_implant_checkpoint or args.implant_checkpoint
    train_implant_name = args.train_implant_name or args.implant_name
    eval_implant_mode = args.eval_implant or train_implant_mode
    eval_implant_checkpoint = args.eval_implant_checkpoint
    eval_implant_name = args.eval_implant_name
    if eval_implant_mode == train_implant_mode and eval_implant_checkpoint is None and eval_implant_name is None:
        eval_implant_checkpoint = train_implant_checkpoint
        eval_implant_name = train_implant_name

    train_implant = build_implant_from_spec(
        mode=train_implant_mode,
        checkpoint=train_implant_checkpoint,
        name=train_implant_name,
        weights_root=args.weights_root,
        device=device,
        diffusion_sampler=args.diffusion_sampler,
        ddim_steps=args.ddim_steps,
        channel=args.channel,
    )
    eval_implant = build_implant_from_spec(
        mode=eval_implant_mode,
        checkpoint=eval_implant_checkpoint,
        name=eval_implant_name,
        weights_root=args.weights_root,
        device=device,
        diffusion_sampler=args.diffusion_sampler,
        ddim_steps=args.ddim_steps,
        channel=args.channel,
    )
    maybe_gpu_warmup(
        device,
        enabled=args.enable_gpu_warmup,
        seconds=args.gpu_warmup_seconds,
        size=args.gpu_warmup_size,
    )

    rate = args.sequence_length / args.channel_length
    batches_per_epoch = max(1, int(args.sample_size / args.batch_size))
    effective_batches = resolve_effective_batches(args)
    overnight_batches_per_epoch = max(1, int(args.sample_size / effective_batches["enc_effective_batch_size"]))
    overnight_eval_batches = (
        max(1, int(args.eval_num_blocks / effective_batches["enc_effective_batch_size"]))
        if int(args.eval_num_blocks) > 0
        else int(args.eval_batches)
    )
    history = []
    train_start = time.perf_counter()

    training_regime = args.training_regime
    if training_regime == "epoch_joint":
        args.training_mode = "joint"
    elif training_regime == "epoch_alternate":
        args.training_mode = "alternate"

    if training_regime == "overfit_fresh_noise":
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate, weight_decay=0.01)
        input_data = generate_data(args.batch_size, args.sequence_length, args.num_symbols, device)
        encoder.train()
        decoder.train()
        for iteration in range(args.num_iterations):
            optimizer.zero_grad()
            encoded_data = encoder(input_data)
            noisy_data = apply_implant(train_implant, encoded_data, ebno_db=args.ebno_db, rate=rate, device=device)
            logits = decoder(noisy_data)
            loss = F.binary_cross_entropy_with_logits(logits, input_data)
            probs = torch.sigmoid(logits)
            ber = compute_ber(probs, input_data)
            bler = compute_bler(probs, input_data)
            loss.backward()
            optimizer.step()
            if iteration % args.log_every == 0 or iteration + 1 == args.num_iterations:
                history.append(
                    {
                        "iteration": iteration + 1,
                        "phase": "overfit_fresh_noise",
                        "train_loss": float(loss.item()),
                        "train_ber": float(ber),
                        "train_bler": float(bler),
                    }
                )
                print(
                    f"iter {iteration + 1}/{args.num_iterations}: "
                    f"loss={history[-1]['train_loss']:.6e}, "
                    f"train_ber={history[-1]['train_ber']:.6e}, "
                    f"train_bler={history[-1]['train_bler']:.6e}",
                    flush=True,
                )
        eval_stats = evaluate_model(
            encoder,
            decoder,
            eval_implant,
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_symbols=args.num_symbols,
            ebno_db=args.ebno_db,
            rate=rate,
            device=device,
        )
        if history:
            history[-1]["eval_ber"] = float(eval_stats["ber"])
            history[-1]["eval_bler"] = float(eval_stats["bler"])
        else:
            history.append(
                {
                    "iteration": args.num_iterations,
                    "phase": "overfit_fresh_noise",
                    "eval_ber": float(eval_stats["ber"]),
                    "eval_bler": float(eval_stats["bler"]),
                }
            )
    elif training_regime == "overnight_alternate":
        encoder_optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        decoder_optimizer = torch.optim.AdamW(
            decoder.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_eval_ber = float("inf")
        evals_since_improvement = 0
        lr_reduction_count = 0

        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            encoder.train()
            decoder.train()

            enc_losses = []
            enc_bers = []
            enc_blers = []
            for _ in range(overnight_batches_per_epoch):
                encoder_optimizer.zero_grad(set_to_none=True)
                running_loss = 0.0
                running_ber = 0.0
                running_bler = 0.0
                for _accum in range(effective_batches["enc_accum_steps"]):
                    input_data = generate_data(
                        effective_batches["enc_micro_batch_size"],
                        args.sequence_length,
                        args.num_symbols,
                        device,
                    )
                    with make_autocast_context(device, enabled=args.amp, amp_dtype=args.amp_dtype):
                        encoded_data = encoder(input_data)
                        noisy_data = apply_implant(
                            train_implant,
                            encoded_data,
                            ebno_db=args.ebno_db,
                            rate=rate,
                            device=device,
                            decoder_training=False,
                            ebno_range=(args.decoder_ebno_offset_low, args.decoder_ebno_offset_high),
                        )
                        logits = decoder(noisy_data)
                        loss = F.binary_cross_entropy_with_logits(logits, input_data)
                    probs = torch.sigmoid(logits)
                    running_loss += float(loss.item())
                    running_ber += compute_ber(probs, input_data)
                    running_bler += compute_bler(probs, input_data)
                    (loss / effective_batches["enc_accum_steps"]).backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip_norm)
                encoder_optimizer.step()
                enc_losses.append(running_loss / effective_batches["enc_accum_steps"])
                enc_bers.append(running_ber / effective_batches["enc_accum_steps"])
                enc_blers.append(running_bler / effective_batches["enc_accum_steps"])

            dec_losses = []
            dec_bers = []
            dec_blers = []
            for _ in range(5 * overnight_batches_per_epoch):
                decoder_optimizer.zero_grad(set_to_none=True)
                running_loss = 0.0
                running_ber = 0.0
                running_bler = 0.0
                for _accum in range(effective_batches["dec_accum_steps"]):
                    input_data = generate_data(
                        effective_batches["dec_micro_batch_size"],
                        args.sequence_length,
                        args.num_symbols,
                        device,
                    )
                    with torch.no_grad():
                        with make_autocast_context(device, enabled=args.amp, amp_dtype=args.amp_dtype):
                            encoded_data = encoder(input_data).detach()
                    noisy_data = apply_implant(
                        train_implant,
                        encoded_data,
                        ebno_db=args.ebno_db,
                        rate=rate,
                        device=device,
                        decoder_training=True,
                        ebno_range=(args.decoder_ebno_offset_low, args.decoder_ebno_offset_high),
                    )
                    with make_autocast_context(device, enabled=args.amp, amp_dtype=args.amp_dtype):
                        logits = decoder(noisy_data)
                        loss = F.binary_cross_entropy_with_logits(logits, input_data)
                    probs = torch.sigmoid(logits)
                    running_loss += float(loss.item())
                    running_ber += compute_ber(probs, input_data)
                    running_bler += compute_bler(probs, input_data)
                    (loss / effective_batches["dec_accum_steps"]).backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip_norm)
                decoder_optimizer.step()
                dec_losses.append(running_loss / effective_batches["dec_accum_steps"])
                dec_bers.append(running_ber / effective_batches["dec_accum_steps"])
                dec_blers.append(running_bler / effective_batches["dec_accum_steps"])

            metrics = {
                "epoch": epoch + 1,
                "phase": "overnight_alternate",
                "encoder_loss": float(np.mean(enc_losses)),
                "encoder_ber": float(np.mean(enc_bers)),
                "encoder_bler": float(np.mean(enc_blers)),
                "decoder_loss": float(np.mean(dec_losses)),
                "decoder_ber": float(np.mean(dec_bers)),
                "decoder_bler": float(np.mean(dec_blers)),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "learning_rate": get_optimizer_lr(encoder_optimizer),
                "lr_reduction_count": lr_reduction_count,
                "eval_loss": None,
                "eval_ber": None,
                "eval_bler": None,
            }

            if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
                eval_stats = evaluate_model(
                    encoder,
                    decoder,
                    eval_implant,
                    eval_batches=overnight_eval_batches,
                    batch_size=effective_batches["enc_effective_batch_size"],
                    sequence_length=args.sequence_length,
                    num_symbols=args.num_symbols,
                    ebno_db=args.ebno_db,
                    rate=rate,
                    device=device,
                    amp=args.amp,
                    amp_dtype=args.amp_dtype,
                )
                metrics["eval_loss"] = float(eval_stats["loss"])
                metrics["eval_ber"] = float(eval_stats["ber"])
                metrics["eval_bler"] = float(eval_stats["bler"])
                if metrics["eval_ber"] < best_eval_ber:
                    best_eval_ber = float(metrics["eval_ber"])
                    evals_since_improvement = 0
                    save_e2e_checkpoint(out_dir / "best_eval.pt", encoder, decoder, args, epoch + 1, metrics)
                else:
                    evals_since_improvement += 1

                if (
                    args.lr_plateau_patience_evals > 0
                    and args.lr_plateau_max_reductions > 0
                    and evals_since_improvement >= args.lr_plateau_patience_evals
                    and lr_reduction_count < args.lr_plateau_max_reductions
                ):
                    current_lr = get_optimizer_lr(encoder_optimizer)
                    new_lr = max(current_lr * args.lr_plateau_factor, args.lr_plateau_min_lr)
                    if new_lr < current_lr:
                        set_optimizer_lr(encoder_optimizer, new_lr)
                        set_optimizer_lr(decoder_optimizer, new_lr)
                        lr_reduction_count += 1
                        evals_since_improvement = 0
                        metrics["learning_rate"] = new_lr
                        metrics["lr_reduction_count"] = lr_reduction_count
                        metrics["lr_reduced"] = True

            if args.save_every > 0 and (((epoch + 1) % args.save_every == 0) or epoch + 1 == args.epochs):
                save_e2e_checkpoint(out_dir / f"epoch_{epoch + 1:03d}.pt", encoder, decoder, args, epoch + 1, metrics)
                save_e2e_checkpoint(out_dir / "latest.pt", encoder, decoder, args, epoch + 1, metrics)

            history.append(metrics)
            print(
                f"epoch {epoch + 1}/{args.epochs}: "
                f"enc_loss={metrics['encoder_loss']:.6e}, "
                f"enc_ber={metrics['encoder_ber']:.6e}, "
                f"enc_bler={metrics['encoder_bler']:.6e}, "
                f"dec_loss={metrics['decoder_loss']:.6e}, "
                f"dec_ber={metrics['decoder_ber']:.6e}, "
                f"dec_bler={metrics['decoder_bler']:.6e}, "
                f"eval_ber={metrics['eval_ber'] if metrics['eval_ber'] is not None else 'skipped'}, "
                f"eval_bler={metrics['eval_bler'] if metrics['eval_bler'] is not None else 'skipped'}",
                flush=True,
            )
    elif args.training_mode == "joint":
        optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate, weight_decay=0.01)

        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            encoder.train()
            decoder.train()
            losses = []
            bers = []
            blers = []
            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                input_data = generate_data(args.batch_size, args.sequence_length, args.num_symbols, device)
                encoded_data = encoder(input_data)
                noisy_data = apply_implant(train_implant, encoded_data, ebno_db=args.ebno_db, rate=rate, device=device)
                logits = decoder(noisy_data)
                loss = F.binary_cross_entropy_with_logits(logits, input_data)
                probs = torch.sigmoid(logits)
                ber = compute_ber(probs, input_data)
                bler = compute_bler(probs, input_data)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                bers.append(float(ber))
                blers.append(float(bler))

            eval_ber = None
            eval_bler = None
            if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
                eval_stats = evaluate_model(
                    encoder,
                    decoder,
                    eval_implant,
                    eval_batches=args.eval_batches,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    num_symbols=args.num_symbols,
                    ebno_db=args.ebno_db,
                    rate=rate,
                    device=device,
                )
                eval_ber = eval_stats["ber"]
                eval_bler = eval_stats["bler"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "phase": "joint",
                    "train_loss": float(np.mean(losses)),
                    "train_ber": float(np.mean(bers)),
                    "train_bler": float(np.mean(blers)),
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "eval_ber": None if eval_ber is None else float(eval_ber),
                    "eval_bler": None if eval_bler is None else float(eval_bler),
                }
            )
            print(
                f"epoch {epoch + 1}/{args.epochs}: "
                f"loss={history[-1]['train_loss']:.6e}, "
                f"train_ber={history[-1]['train_ber']:.6e}, "
                f"train_bler={history[-1]['train_bler']:.6e}, "
                f"eval_ber={history[-1]['eval_ber'] if history[-1]['eval_ber'] is not None else 'skipped'}, "
                f"eval_bler={history[-1]['eval_bler'] if history[-1]['eval_bler'] is not None else 'skipped'}",
                flush=True,
            )
    else:
        encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.01)
        decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=0.01)

        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            encoder.train()
            decoder.train()
            enc_losses = []
            enc_bers = []
            enc_blers = []
            for _ in range(batches_per_epoch):
                encoder_optimizer.zero_grad()
                input_data = generate_data(args.batch_size, args.sequence_length, args.num_symbols, device)
                encoded_data = encoder(input_data)
                noisy_data = apply_implant(train_implant, encoded_data, ebno_db=args.ebno_db, rate=rate, device=device)
                logits = decoder(noisy_data)
                loss = F.binary_cross_entropy_with_logits(logits, input_data)
                probs = torch.sigmoid(logits)
                ber = compute_ber(probs, input_data)
                bler = compute_bler(probs, input_data)
                loss.backward()
                encoder_optimizer.step()
                enc_losses.append(float(loss.item()))
                enc_bers.append(float(ber))
                enc_blers.append(float(bler))

            decoder_batch_size = args.batch_size * args.dec_bs_fac
            dec_losses = []
            dec_bers = []
            dec_blers = []
            for _ in range(5 * batches_per_epoch):
                decoder_optimizer.zero_grad()
                input_data = generate_data(decoder_batch_size, args.sequence_length, args.num_symbols, device)
                encoded_data = encoder(input_data).detach()
                noisy_data = apply_implant(
                    train_implant,
                    encoded_data,
                    ebno_db=args.ebno_db,
                    rate=rate,
                    device=device,
                    decoder_training=True,
                )
                logits = decoder(noisy_data)
                loss = F.binary_cross_entropy_with_logits(logits, input_data)
                probs = torch.sigmoid(logits)
                ber = compute_ber(probs, input_data)
                bler = compute_bler(probs, input_data)
                loss.backward()
                decoder_optimizer.step()
                dec_losses.append(float(loss.item()))
                dec_bers.append(float(ber))
                dec_blers.append(float(bler))

            eval_ber = None
            eval_bler = None
            if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
                eval_stats = evaluate_model(
                    encoder,
                    decoder,
                    eval_implant,
                    eval_batches=args.eval_batches,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    num_symbols=args.num_symbols,
                    ebno_db=args.ebno_db,
                    rate=rate,
                    device=device,
                )
                eval_ber = eval_stats["ber"]
                eval_bler = eval_stats["bler"]

            history.append(
                {
                    "epoch": epoch + 1,
                    "phase": "alternate",
                    "encoder_loss": float(np.mean(enc_losses)),
                    "encoder_ber": float(np.mean(enc_bers)),
                    "encoder_bler": float(np.mean(enc_blers)),
                    "decoder_loss": float(np.mean(dec_losses)),
                    "decoder_ber": float(np.mean(dec_bers)),
                    "decoder_bler": float(np.mean(dec_blers)),
                    "epoch_seconds": time.perf_counter() - epoch_start,
                    "eval_ber": None if eval_ber is None else float(eval_ber),
                    "eval_bler": None if eval_bler is None else float(eval_bler),
                }
            )
            print(
                f"epoch {epoch + 1}/{args.epochs}: "
                f"enc_loss={history[-1]['encoder_loss']:.6e}, "
                f"enc_ber={history[-1]['encoder_ber']:.6e}, "
                f"enc_bler={history[-1]['encoder_bler']:.6e}, "
                f"dec_loss={history[-1]['decoder_loss']:.6e}, "
                f"dec_ber={history[-1]['decoder_ber']:.6e}, "
                f"dec_bler={history[-1]['decoder_bler']:.6e}, "
                f"eval_ber={history[-1]['eval_ber'] if history[-1]['eval_ber'] is not None else 'skipped'}, "
                f"eval_bler={history[-1]['eval_bler'] if history[-1]['eval_bler'] is not None else 'skipped'}",
                flush=True,
            )

    if training_regime in {"epoch_joint", "epoch_alternate"} and args.epochs == 0:
        eval_stats = evaluate_model(
            encoder,
            decoder,
            eval_implant,
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_symbols=args.num_symbols,
            ebno_db=args.ebno_db,
            rate=rate,
            device=device,
        )
        history.append(
            {
                "epoch": 0,
                "phase": "eval_only",
                "eval_ber": float(eval_stats["ber"]),
                "eval_bler": float(eval_stats["bler"]),
            }
        )
        print(
            f"eval_only: eval_ber={history[-1]['eval_ber']:.6e}, "
            f"eval_bler={history[-1]['eval_bler']:.6e}",
            flush=True,
        )

    train_seconds = time.perf_counter() - train_start
    final_eval = {
        key: value
        for key, value in (history[-1] if history else {}).items()
        if key.startswith("eval_")
    }

    summary = {
        "model_type": args.model_type,
        "channel": args.channel,
        "train_implant": getattr(train_implant, "name", train_implant_mode),
        "train_implant_source": train_implant_name or train_implant_checkpoint,
        "eval_implant": getattr(eval_implant, "name", eval_implant_mode),
        "eval_implant_source": eval_implant_name or eval_implant_checkpoint,
        "seed": args.seed,
        "device": str(device),
        "training_regime": training_regime,
        "rate": rate,
        "ebno_db": args.ebno_db,
        "sequence_length": args.sequence_length,
        "channel_length": args.channel_length,
        "epochs": args.epochs,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "dec_bs_fac": args.dec_bs_fac,
        "effective_batches": effective_batches,
        "sample_size": args.sample_size,
        "eval_num_blocks": args.eval_num_blocks,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "decoder_ebno_offset_low": args.decoder_ebno_offset_low,
        "decoder_ebno_offset_high": args.decoder_ebno_offset_high,
        "lr_plateau_patience_evals": args.lr_plateau_patience_evals,
        "lr_plateau_factor": args.lr_plateau_factor,
        "lr_plateau_max_reductions": args.lr_plateau_max_reductions,
        "lr_plateau_min_lr": args.lr_plateau_min_lr,
        "train_seconds": train_seconds,
        "final_eval": final_eval,
        "history": history,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.save_e2e_name:
        checkpoint_path = build_artifact_path(
            root=args.weights_root,
            category="e2e_autoencoders",
            group=args.model_type,
            name=args.save_e2e_name,
            suffix=".pt",
        )
        payload = {
            "model_type": args.model_type,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "config": {
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "channel_length": args.channel_length,
                "enc_num_layers": args.enc_num_layers,
                "dec_num_layers": args.dec_num_layers,
                "hidden_size_gru": args.hidden_size_gru,
                "num_iteration": args.num_iteration,
                "num_iter_ft_cnn": args.num_iter_ft_cnn,
            },
            "summary_path": str(summary_path.resolve()),
        }
        torch.save(payload, checkpoint_path)
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(summary, indent=2))
        register_artifact(
            name=args.save_e2e_name,
            category="e2e_autoencoder",
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            root=args.weights_root,
            extra={
                "model_type": args.model_type,
                "seed": args.seed,
                "train_implant": summary["train_implant"],
                "eval_implant": summary["eval_implant"],
            },
        )
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
