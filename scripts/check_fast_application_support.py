"""Localize analytic-channel errors at the saved surrogate-trained codewords."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from conditional_drifting.baselines.fast_channels import GaussianNetwork, physical_channel
from conditional_drifting.baselines.fast_evaluation import codec, write_json
from conditional_drifting.baselines.official_sources import sha256
from scripts.run_fast_channel_application import gaussian_law
from scripts.run_local_gradient_fidelity import normalized_codebook


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(2)
    path = args.suite_dir / "support_check.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    checks = json.loads((args.suite_dir / "own_codebook_check.json").read_text())
    for name, digest in manifest["adapter_files"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Source changed: {name}")
    if manifest["settings"]["channel"] != "SSPA":
        raise ValueError("This support check uses the eight-real-coordinate SSPA law")
    rows = []
    seed = manifest["settings"]["seed"]+960001
    for row in checks["rows"]:
        name = row["method"]
        ckpt = args.suite_dir / name / "codec.pt"
        if sha256(ckpt) != row["codec_sha256"]:
            raise ValueError("Codec checkpoint changed")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        encoder, decoder, messages = codec("SSPA", args.device, state["seed"])
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        codes = normalized_codebook(encoder, messages)
        torch.testing.assert_close(codes, codes.new_tensor(row["codebook"]), atol=0, rtol=0)
        errors = torch.zeros(len(codes), device=args.device, dtype=torch.long)
        torch.manual_seed(seed)
        # Identical order/batches/RNG to the saved analytic expected_loss evaluation.
        samples, chunk = checks["samples_per_message"], 2048//len(codes)
        for start in range(0, samples, chunk):
            count = min(chunk, samples-start)
            x = codes[:, None, :].expand(len(codes), count, codes.shape[1]).reshape(-1, codes.shape[1])
            predictions = decoder(physical_channel("SSPA", x)).argmax(-1).reshape(len(codes), count)
            errors += (predictions != torch.arange(len(codes), device=args.device)[:, None]).sum(1)
        if errors.sum().item() != row["analytic"]["errors"]:
            raise ValueError("Per-message replay did not reproduce saved analytic error count")
        norms = codes.norm(dim=1)
        z = norms.square().double()/2
        # For iid N(0,1) inputs in eight real coordinates, squared radius is chi-square(8).
        tail = (-z).exp()*sum(z**k/math.factorial(k) for k in range(4))
        result = dict(method=name, input_norms=norms.tolist(), radial_tail_probabilities=tail.tolist(),
            analytic_errors_per_message=errors.tolist(), samples_per_message=samples,
            errors_match_saved_evaluation=True)
        if name == "gaussian":
            gen = args.suite_dir / name / "generator.pt"
            if sha256(gen) != row["generator_sha256"]:
                raise ValueError("Generator changed")
            saved = torch.load(gen, map_location="cpu", weights_only=False)["trainer"]
            network = GaussianNetwork(8, saved["config"]["parameters"]).to(args.device)
            network.load_state_dict(saved["ema"])
            mean, scale = network.distribution(codes)
            truth, _ = gaussian_law("SSPA", codes)
            result.update(exact_learned_mean=mean.tolist(), exact_analytic_mean=truth.tolist(),
                learned_mean_iq_amplitudes=mean.reshape(len(codes), -1, 2).norm(dim=-1).tolist(),
                analytic_mean_iq_amplitudes=truth.reshape(len(codes), -1, 2).norm(dim=-1).tolist(),
                exact_mean_rmse_per_message=(mean-truth).square().mean(1).sqrt().tolist())
        rows.append(result)
        worst = errors.argsort(descending=True)[:3].tolist()
        print(f"[support] {name} total_errors={errors.sum().item()} highest_error_messages={[(i, errors[i].item()) for i in worst]}", flush=True)
    write_json(path, dict(rows=rows, own_codebook_check_sha256=sha256(args.suite_dir / "own_codebook_check.json"),
        script_sha256=sha256(Path(__file__)),
        qualification="Post-hoc localization of the existing evaluation, not a controlled causal intervention"))


if __name__ == "__main__":
    main()
