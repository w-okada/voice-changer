import math
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import commons, modules
from .commons import get_padding
from .modules import (ConvNext2d, HarmonicEmbedder, IMDCTSymExpHead,
                      LoRALinear1d, SnakeFilter, WaveBlock)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

sr2sr = {
    "24k": 24000,
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

class GeneratorVoras(torch.nn.Module):
    def __init__(
        self,
        emb_channels,
        inter_channels,
        gin_channels,
        n_layers,
        sr,
        hop_length,
    ):
        super(GeneratorVoras, self).__init__()
        self.n_layers = n_layers
        self.emb_pitch = HarmonicEmbedder(768, inter_channels, gin_channels, 16, 15)  #   # pitch 256
        self.plinear = LoRALinear1d(inter_channels, inter_channels, gin_channels, r=8)
        self.glinear = weight_norm(nn.Conv1d(gin_channels, inter_channels, 1))
        self.resblocks = nn.ModuleList()
        self.init_linear = LoRALinear1d(emb_channels, inter_channels, gin_channels, r=4)
        for _ in range(self.n_layers):
            self.resblocks.append(WaveBlock(inter_channels, gin_channels, [9] * 2, [1] * 2, [1, 9], 2, r=4))
        self.head = IMDCTSymExpHead(inter_channels, gin_channels, hop_length, padding="center", sample_rate=sr)
        self.post = SnakeFilter(4, 8, 9, 2, eps=1e-5)

    def forward(self, x, pitchf, x_mask, g):
        x = self.init_linear(x, g) + self.plinear(self.emb_pitch(pitchf, g), g) + self.glinear(g)
        for i in range(self.n_layers):
            x = self.resblocks[i](x, x_mask, g)
        x = x * x_mask
        x = self.head(x, g)
        x = self.post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        self.plinear.remove_weight_norm()
        remove_weight_norm(self.glinear)
        for l in self.resblocks:
            l.remove_weight_norm()
        self.init_linear.remove_weight_norm()
        self.head.remove_weight_norm()
        self.post.remove_weight_norm()

    def fix_speaker(self, g):
        self.plinear.fix_speaker(g)
        self.init_linear.fix_speaker(g)
        for l in self.resblocks:
            l.fix_speaker(g)
        self.head.fix_speaker(g)

    def unfix_speaker(self, g):
        self.plinear.unfix_speaker(g)
        self.init_linear.unfix_speaker(g)
        for l in self.resblocks:
            l.unfix_speaker(g)
        self.head.unfix_speaker(g)


class Synthesizer(nn.Module):
    def __init__(
        self,
        segment_size,
        n_fft,
        hop_length,
        inter_channels,
        n_layers,
        spk_embed_dim,
        gin_channels,
        emb_channels,
        sr,
        **kwargs
    ):
        super().__init__()
        if type(sr) == type("strr"):
            sr = sr2sr[sr]
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.inter_channels = inter_channels
        self.n_layers = n_layers
        self.spk_embed_dim = spk_embed_dim
        self.gin_channels = gin_channels
        self.emb_channels = emb_channels
        self.sr = sr

        self.dec = GeneratorVoras(
            emb_channels,
            inter_channels,
            gin_channels,
            n_layers,
            sr,
            hop_length
        )

        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        print(
            "gin_channels:",
            gin_channels,
            "self.spk_embed_dim:",
            self.spk_embed_dim,
            "emb_channels:",
            emb_channels,
        )
        self.speaker = None

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()

    def change_speaker(self, sid: int):
        if self.speaker is not None:
            g = self.emb_g(torch.from_numpy(np.array(self.speaker))).unsqueeze(-1)
            self.dec.unfix_speaker(g)
        g = self.emb_g(torch.from_numpy(np.array(sid))).unsqueeze(-1)
        self.dec.fix_speaker(g)
        self.speaker = sid

    def forward(
        self, phone, phone_lengths, pitch, pitchf, ds
        ):
        g = self.emb_g(ds).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        x_slice, ids_slice = commons.rand_slice_segments(
            x, phone_lengths, self.segment_size
        )
        pitchf_slice = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        mask_slice = commons.slice_segments(x_mask, ids_slice, self.segment_size)
        o = self.dec(x_slice, pitchf_slice, mask_slice, g)
        return o, ids_slice, x_mask, g

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        x = torch.transpose(phone, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(phone.dtype)
        o = self.dec((x * x_mask)[:, :, :max_len], nsff0, x_mask, g)
        return o, x_mask, (None, None, None, None)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, gin_channels, upsample_rates, final_dim=256, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        self.init_kernel_size = upsample_rates[-1] * 3
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        N = len(upsample_rates)
        self.init_conv = norm_f(Conv2d(1, final_dim // (2 ** (N - 1)), (self.init_kernel_size, 1), (upsample_rates[-1], 1)))
        self.convs = nn.ModuleList()
        for i, u in enumerate(upsample_rates[::-1][1:], start=1):
            self.convs.append(
                ConvNext2d(
                    final_dim // (2 ** (N - i)),
                    final_dim // (2 ** (N - i - 1)),
                    gin_channels,
                    (u*3, 1),
                    (u, 1),
                    4,
                    r=2 + i//2
                )
            )
        self.conv_post = weight_norm(Conv2d(final_dim, 1, (3, 1), (1, 1)))

    def forward(self, x, g):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (n_pad, 0), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        x = torch.flip(x, dims=[2])
        x = F.pad(x, [0, 0, 0, self.init_kernel_size - 1], mode="constant")
        x = self.init_conv(x)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = torch.flip(x, dims=[2])
        fmap.append(x)

        for i, l in enumerate(self.convs):
            x = l(x, g)
            fmap.append(x)

        x = F.pad(x, [0, 0, 2, 0], mode="constant")
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, upsample_rates, gin_channels, periods=[2, 3, 5, 7, 11, 17], **kwargs):
        super(MultiPeriodDiscriminator, self).__init__()

        discs = [
            DiscriminatorP(i, gin_channels, upsample_rates, use_spectral_norm=False) for i in periods
        ]
        self.ups = np.prod(upsample_rates)
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat, g):
        fmap_rs = []
        fmap_gs = []
        y_d_rs = []
        y_d_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y, g)
            y_d_g, fmap_g = d(y_hat, g)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
