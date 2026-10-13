"""Bounded receiver development on the physical OFDM channel, before surrogate comparison."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from conditional_drifting.application_channels import OFDMChannel, as_complex, as_real
from conditional_drifting.ofdm_application import Constellation, PilotReceiver, evaluate_receiver
from conditional_drifting.baselines.fast_evaluation import save, sync, write_json
from conditional_drifting.baselines.official_sources import sha256
from conditional_drifting.sspa_trajectory import rng_state, restore_rng
from conditional_drifting.training import set_seed
from scripts.run_fast_channel_comparison import ensure_manifest


class FrequencyReceiver(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.data = list(channel.data_indices)
        self.input = nn.Conv1d(3, 32, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(1, 32), nn.SiLU(),
            nn.Conv1d(32, 32, 3, padding=d, dilation=d), nn.GroupNorm(1, 32), nn.SiLU(),
            nn.Conv1d(32, 32, 3, padding=d, dilation=d)) for d in (1, 2, 4, 8)])
        self.output = nn.Conv1d(32, 4, 1)
        mask = torch.zeros(1, 1, 64)
        mask[:, :, list(channel.pilot_indices)] = 1
        self.register_buffer("mask", mask)

    def forward(self, y):
        iq = y.reshape(len(y), 64, 2).transpose(1, 2)
        hidden = self.input(torch.cat((iq, self.mask.expand(len(y), -1, -1)), 1))
        for block in self.blocks:
            hidden = hidden+block(hidden)
        return self.output(hidden).transpose(1, 2)[:, self.data]


class PilotResidualReceiver(PilotReceiver):
    def __init__(self, channel, device):
        super().__init__(channel, channel.esno_db, device)
        self.correction = nn.Sequential(nn.Linear(6*64, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 4*56)).to(device)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, y):
        yc = as_complex(y)
        h = yc[:, list(self.channel.pilot_indices)] @ self.weights.T
        z = yc*h.conj()/(h.abs().square()+self.effective_noise)
        base = -(z[:, list(self.channel.data_indices), None]-self.constellation()[None, None]).abs().square()
        correction = self.correction(torch.cat((y, as_real(h), as_real(z)), -1)).reshape_as(base)
        return base+correction


def run(args):
    torch.set_num_threads(2)
    c = OFDMChannel(activity=.1 if args.condition == "interference" else 0.)
    paths = [Path(__file__), ROOT/"conditional_drifting/application_channels.py",
        ROOT/"conditional_drifting/ofdm_application.py", ROOT/"conditional_drifting/baselines/fast_evaluation.py",
        ROOT/"conditional_drifting/sspa_trajectory.py", ROOT/"conditional_drifting/training.py"]
    settings = dict(channel=c.description(), architecture=args.architecture, seed=args.seed,
        updates=args.updates, batch=256, learning_rate=.001, train_esno_db=10.,
        device=args.device, torch=str(torch.__version__), eval_blocks=16384,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        scope="Receiver development on true channel only; not a surrogate comparison",
        statistical_prior=args.architecture == "pilot_residual")
    ensure_manifest(args.out_dir/"manifest.json", json.loads(json.dumps(settings)), args.resume)
    set_seed(args.seed)
    model = (FrequencyReceiver(c).to(args.device) if args.architecture == "conv" else PilotResidualReceiver(c,args.device))
    constellation = Constellation(False).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    labels_rng = torch.Generator(device=args.device).manual_seed(args.seed+100000)
    path = args.out_dir/"receiver.pt"
    start, seconds, trace = 0, 0., []
    if path.exists():
        if not args.resume:
            raise FileExistsError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        labels_rng.set_state(state["labels_rng"].cpu())
        restore_rng(state["rng"])
        start, seconds, trace = state["update"], state["seconds"], state["trace"]
    for update in range(start, args.updates):
        sync(args.device)
        before = time.perf_counter()
        labels = torch.randint(4, (256, 56), device=args.device, generator=labels_rng)
        loss = F.cross_entropy(model(c.sample(c.waveform(labels))).flatten(0, 1), labels.flatten())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        sync(args.device)
        seconds += time.perf_counter()-before
        if (update+1) % 100 == 0 or update+1 == args.updates:
            trace.append(dict(update=update+1, loss=loss.item(), seconds=seconds))
        if (update+1) % 500 == 0 or update+1 == args.updates:
            save(path, dict(model=model.state_dict(),optimizer=optimizer.state_dict(),labels_rng=labels_rng.get_state(),
                rng=rng_state(),update=update+1,seconds=seconds,trace=trace))
            print(f"[receiver development] {args.architecture} {args.condition} update={update+1} loss={loss.item():.5g}",flush=True)
    result_path=args.out_dir/"results.json"
    if args.resume and result_path.exists():
        if json.loads(result_path.read_text())["checkpoint_sha256"] != sha256(path):
            raise ValueError("Completed receiver checkpoint changed")
        return
    curves=evaluate_receiver(c,model.eval(),constellation,args.device,16384,512,609401,[0.,5.,10.,15.,20.])
    conventional=[evaluate_receiver(c,PilotReceiver(c,snr,args.device),constellation,args.device,
        16384,512,609401,[snr])[0] for snr in (0.,5.,10.,15.,20.)]
    write_json(result_path,dict(manifest_sha256=sha256(args.out_dir/"manifest.json"),checkpoint_sha256=sha256(path),
        train_seconds=seconds,parameters=sum(p.numel() for p in model.parameters()),trace=trace,
        curves=curves,conventional=conventional))
    print([(row["esno_db"],row["groups"]["all"]["ber"]) for row in curves],flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture",choices=("conv","pilot_residual"),default="conv")
    p.add_argument("--condition",choices=("clean","interference"),default="clean")
    p.add_argument("--seed",type=int,default=409401)
    p.add_argument("--updates",type=int,default=5000)
    p.add_argument("--device",default="cpu")
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.updates < 1:
        raise ValueError("Positive update budget required")
    run(args)
