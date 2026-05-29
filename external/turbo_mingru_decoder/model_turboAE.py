from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import arange
from numpy.random import mtrand


class TurboConfig:
    num_iteration = 6
    code_rate_k = 1
    dec_num_unit = 100
    dec_kernel_size = 5
    enc_num_unit = 100
    enc_kernel_size = 5
    num_iter_ft = 10
    num_iter_ft_cnn = 10
    num_iter_ft_mingru = 10

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Interleaver(torch.nn.Module):
    """TurboAE interleaver compatible with the original local baseline."""

    def __init__(self, config):
        super().__init__()
        seed = np.random.randint(0, 2**31 - 1)
        rand_gen = mtrand.RandomState(seed)
        p_array = rand_gen.permutation(arange(config.block_len))
        self.set_parray(p_array)

    def set_parray(self, p_array):
        p = torch.as_tensor(p_array, dtype=torch.long)
        rev = torch.empty_like(p)
        rev[p] = torch.arange(p.numel(), dtype=torch.long)
        self.register_buffer("p_array", p)
        self.register_buffer("reverse_p_array", rev)

    def _permute(self, x, idx):
        return x.index_select(dim=1, index=idx)

    def interleave(self, x):
        return self._permute(x, self.p_array)

    def deinterleave(self, x):
        return self._permute(x, self.reverse_p_array)


class ModuleLambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def build_encoder_block(num_layer, in_channels, out_channels, kernel_size, activation="elu"):
    layers = [ModuleLambda(lambda x: torch.transpose(x, 1, 2))]
    for idx in range(num_layer):
        layers.append(
            nn.Conv1d(
                in_channels=in_channels if idx == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                dilation=1,
                groups=1,
                bias=True,
            )
        )
        layers.append(ModuleLambda(lambda x: getattr(F, activation)(x)))
    layers.append(ModuleLambda(lambda x: torch.transpose(x, 1, 2)))
    return nn.Sequential(*layers)


class SaturatedSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < -1] = 0
        grad_input[input_tensor > 1] = 0
        return grad_input


class ENC_CNNTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super().__init__()
        self.config = config
        self.interleaver = interleaver
        self.enc_cnn_1 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.enc_num_unit, config.enc_kernel_size)
        self.enc_cnn_2 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.enc_num_unit, config.enc_kernel_size)
        self.enc_linear_1 = nn.Linear(config.enc_num_unit, 1)
        self.enc_linear_2 = nn.Linear(config.enc_num_unit, 1)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2).float()
        x_sys = self.enc_cnn_1(inputs)
        x_sys = F.elu(self.enc_linear_1(x_sys))
        x_p1 = self.enc_cnn_2(self.interleaver.interleave(inputs))
        x_p1 = F.elu(self.enc_linear_2(x_p1))
        x_tx = torch.cat([x_sys, x_p1], dim=2)
        codes = self.power_constraint(x_tx)
        return codes.squeeze(dim=2)


class DEC_CNNTurbo(nn.Module):
    def __init__(self, config, interleaver):
        super().__init__()
        self.config = config
        self.interleaver = interleaver
        ft_cnn = config.num_iter_ft_cnn
        self.dec1_cnns = torch.nn.ModuleList(
            [build_encoder_block(config.dec_num_layer, 2 + ft_cnn, config.dec_num_unit, config.dec_kernel_size) for _ in range(config.num_iteration)]
        )
        self.dec2_cnns = torch.nn.ModuleList(
            [build_encoder_block(config.dec_num_layer, 2 + ft_cnn, config.dec_num_unit, config.dec_kernel_size) for _ in range(config.num_iteration)]
        )
        self.dec1_outputs = torch.nn.ModuleList([torch.nn.Linear(config.dec_num_unit, ft_cnn) for _ in range(config.num_iteration)])
        self.dec2_outputs = torch.nn.ModuleList(
            [torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else ft_cnn) for idx in range(config.num_iteration)]
        )

    def forward(self, received):
        device = next(self.parameters()).device
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2).to(device).float()
        config = self.config
        r_sys = received[:, :, 0].view((bs, config.block_len, 1))
        r_sys_int = self.interleaver.interleave(r_sys)
        r_par = received[:, :, 1].view((bs, config.block_len, 1))
        r_par_deint = self.interleaver.deinterleave(r_par)
        prior = torch.zeros((bs, config.block_len, config.num_iter_ft_cnn), device=device)

        for idx in range(config.num_iteration - 1):
            _, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[idx], self.dec1_outputs[idx])
            x_plr_int = self.interleaver.interleave(x_plr - prior)
            _, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[idx], self.dec2_outputs[idx])
            prior = self.interleaver.deinterleave(x_plr - x_plr_int)

        _, x_plr = self._turbo_decoder_step(r_sys, r_par_deint, prior, self.dec1_cnns[-1], self.dec1_outputs[-1])
        x_plr_int = self.interleaver.interleave(x_plr - prior)
        _, x_plr = self._turbo_decoder_step(r_sys_int, r_par, x_plr_int, self.dec2_cnns[-1], self.dec2_outputs[-1])
        final = self.interleaver.deinterleave(x_plr)
        return final.squeeze(dim=2)

    def _turbo_decoder_step(self, r_sys, r_par, prior, cnn, linear):
        x_this_dec = torch.cat([r_sys, r_par, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_plr = linear(x_dec)
        return x_dec, x_plr


class ENC_CNNTurbo_serial(nn.Module):
    def __init__(self, config, interleaver):
        super().__init__()
        self.config = config
        self.interleaver = interleaver
        ft_cnn = config.num_iter_ft_cnn
        self.enc_cnn_1 = build_encoder_block(config.enc_num_layer, config.code_rate_k, config.enc_num_unit, config.enc_kernel_size)
        self.enc_cnn_2 = build_encoder_block(config.enc_num_layer, config.code_rate_k * ft_cnn, config.enc_num_unit, config.enc_kernel_size)
        self.enc_linear_1 = nn.Linear(config.enc_num_unit, ft_cnn)
        self.enc_linear_2 = nn.Linear(config.enc_num_unit, 2)

    def power_constraint(self, x_input):
        this_mean = torch.mean(x_input)
        this_std = torch.std(x_input)
        return (x_input - this_mean) / this_std

    def forward(self, inputs):
        inputs = inputs.unsqueeze(dim=2)
        inputs = 2.0 * inputs - 1.0
        x_sys = self.enc_cnn_1(inputs)
        x_sys = F.elu(self.enc_linear_1(x_sys))
        x_sys = SaturatedSTE.apply(x_sys)
        x_p1 = self.enc_cnn_2(self.interleaver.interleave(x_sys))
        out = F.elu(self.enc_linear_2(x_p1))
        codes = self.power_constraint(out)
        return codes.squeeze(dim=2)


class DEC_CNNTurbo_serial(nn.Module):
    def __init__(self, config, interleaver):
        super().__init__()
        self.config = config
        self.interleaver = interleaver
        ft_cnn = config.num_iter_ft_cnn
        self.dec1_cnns = torch.nn.ModuleList(
            [build_encoder_block(config.dec_num_layer, 2 + ft_cnn, config.dec_num_unit, config.dec_kernel_size) for _ in range(config.num_iteration)]
        )
        self.dec2_cnns = torch.nn.ModuleList(
            [build_encoder_block(config.dec_num_layer, ft_cnn, config.dec_num_unit, config.dec_kernel_size) for _ in range(config.num_iteration)]
        )
        self.dec1_outputs = torch.nn.ModuleList([torch.nn.Linear(config.dec_num_unit, ft_cnn) for _ in range(config.num_iteration)])
        self.dec2_outputs = torch.nn.ModuleList(
            [torch.nn.Linear(config.dec_num_unit, 1 if idx == config.num_iteration - 1 else ft_cnn) for idx in range(config.num_iteration)]
        )

    def forward(self, received):
        device = next(self.parameters()).device
        bs = received.size(0)
        received = received.view(received.size(0), -1, 2).to(device=device, dtype=torch.float32)
        prior = torch.zeros((bs, self.config.block_len, self.config.num_iter_ft_cnn), device=device)

        for idx in range(self.config.num_iteration):
            _, x_plr1 = self._turbo_decoder_step1(received, prior, self.dec1_cnns[idx], self.dec1_outputs[idx])
            x_plr1_ex = x_plr1 - prior
            x_plr1_ex_int = self.interleaver.deinterleave(x_plr1_ex)
            _, x_plr2 = self._turbo_decoder_step2(x_plr1_ex_int, self.dec2_cnns[idx], self.dec2_outputs[idx])
            x_plr2_ex = x_plr2 - x_plr1_ex_int
            prior = self.interleaver.interleave(x_plr2_ex)

        return torch.sigmoid(x_plr2).squeeze(dim=2)

    def _turbo_decoder_step1(self, input_tensor, prior, cnn, linear):
        x_this_dec = torch.cat([input_tensor, prior], dim=2)
        x_dec = cnn(x_this_dec)
        x_plr = linear(x_dec)
        return x_dec, x_plr

    def _turbo_decoder_step2(self, input_tensor, cnn, linear):
        x_dec = cnn(input_tensor)
        x_plr = linear(x_dec)
        return x_dec, x_plr


class ENC_GRUTurbo(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ImportError("The vendored TurboAE baseline includes only CNN TurboAE classes; use --model-type cnn_turbo.")


class DEC_CNNTurboXminGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise ImportError("The vendored TurboAE baseline includes only CNN TurboAE classes; use --model-type cnn_turbo.")

