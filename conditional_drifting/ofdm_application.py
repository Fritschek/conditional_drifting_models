"""Joint-block receiver training and physical evaluation for the OFDM pilot."""

import math
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

from .application_channels import as_complex, as_real
from .baselines.fast_evaluation import save, sync
from .sspa_trajectory import rng_state, restore_rng, isolated_rng
from .training import set_seed
from .metrics import conditional_anchor_swd, conditional_anchor_cov_fro


class BlockReceiver(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.data = len(channel.data_indices)
        self.net = nn.Sequential(nn.Linear(channel.n, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 4*self.data))

    def forward(self, y):
        return self.net(y).reshape(len(y), self.data, 4)


class Constellation(nn.Module):
    def __init__(self, learned):
        super().__init__()
        points = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])/math.sqrt(2)
        self.points = nn.Parameter(points, requires_grad=learned)

    def forward(self):
        points = self.points-self.points.mean(0)
        return torch.view_as_complex((points/points.square().sum(-1).mean().clamp_min(1e-12).sqrt()).contiguous())


class PilotReceiver(nn.Module):
    """Known second-order channel statistics; no realization or activity labels."""
    def __init__(self, channel, esno_db, device):
        super().__init__()
        self.channel = channel
        cov = channel.frequency_covariance(device)
        pilots = list(channel.pilot_indices)
        noise = 10**(-esno_db/10)+channel.activity*channel.interference_power
        pp = cov[pilots][:, pilots]+noise*torch.eye(len(pilots), device=device)
        weights = torch.linalg.solve(pp, cov[pilots]).mH
        self.register_buffer("weights", weights)
        self.register_buffer("effective_noise", noise+(1-(weights*cov[:, pilots].conj()).sum(-1).real).clamp_min(0))
        self.constellation = Constellation(False).to(device)

    def forward(self, y):
        yc = as_complex(y)
        h = yc[:, list(self.channel.pilot_indices)] @ self.weights.T
        z = yc*h.conj()/(h.abs().square()+self.effective_noise)
        return -(z[:, list(self.channel.data_indices), None]-self.constellation()[None, None]).abs().square()


def train_receiver(channel, sampler, path, device, seed, updates, batch, save_every,
                   resume, learned_constellation=False):
    set_seed(seed)
    receiver = BlockReceiver(channel).to(device)
    constellation = Constellation(learned_constellation).to(device)
    parameters = list(receiver.parameters()) + ([constellation.points] if learned_constellation else [])
    optimizer = torch.optim.Adam(parameters, lr=.001)
    labels_rng = torch.Generator(device=device).manual_seed(seed+100000)
    start, seconds, trace = 0, 0., []
    identity = dict(channel=channel.description(), seed=seed, batch=batch, updates=updates,
                    learned_constellation=learned_constellation)
    if Path(path).exists():
        if not resume:
            raise FileExistsError(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Receiver checkpoint configuration mismatch")
        receiver.load_state_dict(state["receiver"])
        constellation.load_state_dict(state["constellation"])
        optimizer.load_state_dict(state["optimizer"])
        labels_rng.set_state(state["labels_rng"].cpu())
        restore_rng(state["rng"])
        start, seconds, trace = state["update"], state["seconds"], state["trace"]
    for update in range(start, updates):
        sync(device)
        before = time.perf_counter()
        labels = torch.randint(4, (batch, len(channel.data_indices)), generator=labels_rng, device=device)
        x = channel.waveform(labels, constellation())
        loss = F.cross_entropy(receiver(sampler(x)).flatten(0, 1), labels.flatten())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        sync(device)
        seconds += time.perf_counter()-before
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite receiver loss")
        if (update+1) % 100 == 0 or update+1 == updates:
            trace.append(dict(update=update+1, loss=loss.item(), seconds=seconds))
        if (update+1) % save_every == 0 or update+1 == updates:
            save(path, dict(identity=identity, receiver=receiver.state_dict(), constellation=constellation.state_dict(),
                optimizer=optimizer.state_dict(), labels_rng=labels_rng.get_state(), rng=rng_state(),
                update=update+1, seconds=seconds, trace=trace))
            print(f"[receiver] {Path(path).parent.name} update={update+1} loss={loss.item():.5g}", flush=True)
    return receiver.eval(), constellation, dict(seconds=seconds, trace=trace,
        constellation=as_real(constellation()).detach().cpu().tolist(),
        parameters=sum(p.numel() for p in receiver.parameters()))


@torch.no_grad()
def evaluate_receiver(channel, receiver, constellation, device, blocks, batch, seed, esnos):
    rows = []
    for esno in esnos:
        # Separate streams make physical draws identical across different codecs.
        labels_rng = torch.Generator(device=device).manual_seed(seed)
        channel_rng = torch.Generator(device=device).manual_seed(seed+1)
        stats = {name: dict(blocks=0, block_errors=0, symbol_errors=0, bit_errors=0,
                           squared_block_bit_errors=0) for name in ("all", "inactive", "active")}
        for start in range(0, blocks, batch):
            size = min(batch, blocks-start)
            labels = torch.randint(4, (size, len(channel.data_indices)), device=device, generator=labels_rng)
            x = channel.waveform(labels, constellation())
            y, state = channel.sample(x, esno_db=esno, generator=channel_rng, return_state=True)
            prediction = receiver(y).argmax(-1)
            symbol_errors = (prediction != labels).sum(-1)
            bits = (((prediction ^ labels) & 1)+(((prediction ^ labels) >> 1) & 1)).sum(-1)
            for name, mask in (("all", torch.ones(size, dtype=torch.bool, device=device)),
                               ("active", state["activity"]), ("inactive", ~state["activity"])):
                row = stats[name]
                row["blocks"] += mask.sum().item()
                row["block_errors"] += (symbol_errors[mask] > 0).sum().item()
                row["symbol_errors"] += symbol_errors[mask].sum().item()
                row["bit_errors"] += bits[mask].sum().item()
                row["squared_block_bit_errors"] += bits[mask].double().square().sum().item()
        for row in stats.values():
            b, d = row["blocks"], len(channel.data_indices)
            row.update(symbol_trials=b*d, bit_trials=b*d*2,
                       ser=row["symbol_errors"]/(b*d) if b else None,
                       ber=row["bit_errors"]/(b*d*2) if b else None,
                       bler=row["block_errors"]/b if b else None)
            row["ber_mc_se_blocks"] = math.sqrt(max(0., row["squared_block_bit_errors"]-
                row["bit_errors"]**2/b)/(b*(b-1)))/(2*d) if b > 1 else None
            row["zero_block_error_upper95"] = -math.expm1(math.log(.05)/b) if b and row["block_errors"] == 0 else None
        rows.append(dict(esno_db=esno, groups=stats))
    return rows


@torch.no_grad()
def block_metrics(channel, sampler, device, seed, anchors, samples, projections):
    with isolated_rng(seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        labels = torch.randint(4, (anchors, len(channel.data_indices)), device=device, generator=generator)
        x = channel.waveform(labels)
        repeated = x.repeat_interleave(samples, 0)
        truth = channel.sample(repeated).reshape(anchors, samples, -1)
        floor = channel.sample(repeated).reshape_as(truth)
        learned = torch.cat([sampler(v) for v in repeated.split(512)]).reshape_as(truth)
        pc = [2*i+j for i in channel.pilot_indices for j in (0, 1)]
        dc = [2*i+j for i in channel.data_indices for j in (0, 1)]
        def covariance(v):
            centered = v-v.mean(1, keepdim=True)
            return centered.transpose(-1, -2) @ centered/(samples-1)
        cov, cov_l, cov_f = [covariance(v) for v in (truth, learned, floor)]
        def cross_error(v):
            return (v[:, pc][:, :, dc]-cov[:, pc][:, :, dc]).square().sum((-1, -2)).sqrt().mean().item()
        def energy(v):
            return as_complex(v).abs().square().mean(-1).flatten()
        # Thresholds are derived from the first analytic cloud, not selected to favor a method.
        thresholds = torch.quantile(energy(truth), truth.new_tensor([.9, .99]))
        return dict(anchors=anchors, samples=samples, projections=projections,
            swd=conditional_anchor_swd(truth, learned, num_projections=projections, seed=seed),
            swd_floor=conditional_anchor_swd(truth, floor, num_projections=projections, seed=seed),
            covariance_error=conditional_anchor_cov_fro(truth, learned),
            covariance_floor=conditional_anchor_cov_fro(truth, floor),
            pilot_data_covariance_error=cross_error(cov_l), pilot_data_covariance_floor=cross_error(cov_f),
            energy_thresholds=thresholds.tolist(),
            learned_exceedance=[(energy(learned)>v).float().mean().item() for v in thresholds],
            analytic_exceedance=[(energy(floor)>v).float().mean().item() for v in thresholds])
