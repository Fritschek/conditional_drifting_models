"""Resumable frequency-structured receiver training after the plain-MLP gate."""

from pathlib import Path
import time

import torch
import torch.nn.functional as F

from conditional_drifting.ofdm_application import Constellation
from conditional_drifting.baselines.fast_evaluation import save, sync
from conditional_drifting.sspa_trajectory import rng_state, restore_rng
from conditional_drifting.training import set_seed
from scripts.run_ofdm_receiver_development import FrequencyReceiver


def train_receiver(channel, sampler, path, device, seed, updates, batch, save_every,
                   resume, learned_constellation=False):
    set_seed(seed)
    receiver = FrequencyReceiver(channel).to(device)
    constellation = Constellation(learned_constellation).to(device)
    parameters = list(receiver.parameters())+([constellation.points] if learned_constellation else [])
    optimizer = torch.optim.Adam(parameters, lr=.001)
    labels_rng = torch.Generator(device=device).manual_seed(seed+100000)
    start, seconds, trace = 0, 0., []
    identity = dict(channel=channel.description(), seed=seed, batch=batch, updates=updates,
        learned_constellation=learned_constellation, receiver="frequency_resnet32_dilations1_2_4_8")
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
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
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
            print(f"[frequency receiver] {Path(path).parent.name} update={update+1} loss={loss.item():.5g}", flush=True)
    return receiver.eval(), constellation, dict(seconds=seconds, trace=trace,
        constellation=torch.view_as_real(constellation()).detach().cpu().tolist(),
        parameters=sum(p.numel() for p in receiver.parameters()), architecture=identity["receiver"])
