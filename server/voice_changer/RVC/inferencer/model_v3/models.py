import math
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import attentions, commons, modules
from .commons import get_padding, init_weights
from .modules import (CausalConvTranspose1d, ConvNext2d, DilatedCausalConv1d,
                      LoRALinear1d, ResBlock1, WaveConv1D)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        emb_channels: int,
        gin_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: int,
        f0: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.emb_channels = emb_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.emb_phone = nn.Linear(emb_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.emb_g = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, gin_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, phone, pitch, lengths, g):
        if pitch == None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask, g)
        x = self.proj(x)

        return x, None, x_mask


class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def forward(self, f0, upp):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (
                    idx + 2
                )  # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
            rad_values = (f0_buf / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
            rand_ini = torch.rand(
                f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
            )
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            tmp_over_one = torch.cumsum(rad_values, 1)  # % 1  #####%1意味着后面的cumsum无法再优化
            tmp_over_one *= upp
            tmp_over_one = F.interpolate(
                tmp_over_one.transpose(2, 1),
                scale_factor=upp,
                mode="linear",
                align_corners=True,
            ).transpose(2, 1)
            rad_values = F.interpolate(
                rad_values.transpose(2, 1), scale_factor=upp, mode="nearest"
            ).transpose(
                2, 1
            )  #######
            tmp_over_one %= 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sine_waves = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
            sine_waves = sine_waves * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=upp, mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        gin_channels,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        is_half=True,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Conv1d(gin_channels, harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp=None):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_raw = torch.transpose(sine_wavs, 1, 2).to(device=x.device, dtype=x.dtype)
        return sine_raw, None, None  # noise, uv


class GeneratorNSF(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
        harmonic_num=16,
        is_half=False,
    ):
        super(GeneratorNSF, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsample_rates = upsample_rates

        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, gin_channels=gin_channels, harmonic_num=harmonic_num, is_half=is_half
        )
        self.gpre = Conv1d(gin_channels, initial_channel, 1)
        self.conv_pre = ResBlock1(initial_channel, upsample_initial_channel, gin_channels, [7] * 5, [1] * 5, [1, 2, 4, 8, 1], 1, 2)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        c_cur = upsample_initial_channel
        for i, u in enumerate(upsample_rates):
            c_pre = c_cur
            c_cur = c_pre // 2
            self.ups.append(
                CausalConvTranspose1d(
                    c_pre,
                    c_pre,
                    kernel_rate=3,
                    stride=u,
                    groups=c_pre,
                )
            )
            self.resblocks.append(ResBlock1(c_pre, c_cur, gin_channels, [11] * 5, [1] * 5, [1, 2, 4, 8, 1], 1, r=2))
        self.conv_post = DilatedCausalConv1d(c_cur, 1, 5, stride=1, groups=1, dilation=1, bias=False)
        self.noise_convs = nn.ModuleList()
        self.noise_pre = LoRALinear1d(1 + harmonic_num, c_pre, gin_channels, r=2+harmonic_num)
        for i, u in enumerate(upsample_rates[::-1]):
            c_pre = c_pre * 2
            c_cur = c_cur * 2
            if i + 1 < len(upsample_rates):
                self.noise_convs.append(DilatedCausalConv1d(c_cur, c_pre, kernel_size=u*3, stride=u, groups=c_cur, dilation=1))
            else:
                self.noise_convs.append(DilatedCausalConv1d(c_cur, initial_channel, kernel_size=u*3, stride=u, groups=math.gcd(c_cur, initial_channel), dilation=1))
        self.upp = np.prod(upsample_rates)

    def forward(self, x, x_mask, f0f, g):
        har_source, noi_source, uv = self.m_source(f0f, self.upp)
        har_source = self.noise_pre(har_source, g)
        x_sources = [har_source]
        for c in self.noise_convs:
            har_source = c(har_source)
            x_sources.append(har_source)

        x = x + x_sources[-1]
        x = x + self.gpre(g)
        x = self.conv_pre(x, x_mask, g)
        for i, u in enumerate(self.upsample_rates):
            x_mask = torch.repeat_interleave(x_mask, u, 2)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            x = self.resblocks[i](x + x_sources[-i-2], x_mask, g)

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        if x_mask is not None:
            x *= x_mask
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.noise_pre)
        remove_weight_norm(self.conv_post)


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class SynthesizerTrnMs256NSFSid(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        emb_channels,
        sr,
        **kwargs
    ):
        super().__init__()
        if type(sr) == type("strr"):
            sr = sr2sr[sr]
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.emb_channels = emb_channels
        self.sr = sr
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim

        self.emb_pitch = nn.Embedding(256, emb_channels)  # pitch 256
        self.dec = GeneratorNSF(
            emb_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
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

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()

    def forward(
        self, phone, phone_lengths, pitch, pitchf, y, y_lengths, ds
    ):  # 这里ds是id，[bs,1]
        # print(1,pitch.shape)#[bs,t]
        g = self.emb_g(ds).unsqueeze(-1)  # [b, 256, 1]##1是t，广播的
        # m_p, _, x_mask = self.enc_p(phone, pitch, phone_lengths, g)
        # z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        # z_p = self.flow(m_p * x_mask, x_mask, g=g)

        x = phone + self.emb_pitch(pitch)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(
            phone.dtype
        )

        m_p_slice, ids_slice = commons.rand_slice_segments(
            x, phone_lengths, self.segment_size
        )
        # print(-1,pitchf.shape,ids_slice,self.segment_size,self.hop_length,self.segment_size//self.hop_length)
        pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
        mask_slice = commons.slice_segments(x_mask, ids_slice, self.segment_size)
        # print(-2,pitchf.shape,z_slice.shape)
        o = self.dec(m_p_slice, mask_slice, pitchf, g)
        return o, ids_slice, x_mask, g

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        x = phone + self.emb_pitch(pitch)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(
            phone.dtype
        )
        o = self.dec((x * x_mask)[:, :, :max_len], x_mask, nsff0, g)
        return o, x_mask, (None, None, None, None)


class DiscriminatorS(torch.nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            filter_channels: int,
            gin_channels: int,
            n_heads: int,
            n_layers: int,
            kernel_size: int,
            p_dropout: int,
            ):
        super(DiscriminatorS, self).__init__()
        self.convs = WaveConv1D(2, hidden_channels, gin_channels, [10, 7, 7, 7, 5, 3, 3], [5, 4, 4, 4, 3, 2, 2], [1] * 7, hidden_channels // 2, False)
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, gin_channels, n_heads, n_layers//2, kernel_size, p_dropout
        )
        self.cross = weight_norm(torch.nn.Conv1d(gin_channels, hidden_channels, 1, 1))
        self.conv_post = weight_norm(torch.nn.Conv1d(hidden_channels, 1, 3, 1, padding=get_padding(5, 1)))

    def forward(self, x, g):
        x = self.convs(x)
        x_mask = torch.ones([x.shape[0], 1, x.shape[2]], device=x.device, dtype=x.dtype)
        x = self.encoder(x, x_mask, g)
        fmap = [x]
        x = x + x * self.cross(g)
        y = self.conv_post(x)
        return y, fmap


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
                    r=2
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

        for i, l in enumerate(self.convs):
            x = l(x, g)
            if i >= 1:
                fmap.append(x)

        x = F.pad(x, [0, 0, 2, 0], mode="constant")
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, upsample_rates, gin_channels, periods=[2, 3, 5, 7, 11, 17], **kwargs):
        super(MultiPeriodDiscriminator, self).__init__()

        # discs = [DiscriminatorS(hidden_channels, filter_channels, gin_channels, n_heads, n_layers, kernel_size, p_dropout)]
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
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, g)
            y_d_g, fmap_g = d(y_hat, g)
            # for j in range(len(fmap_r)):
            #     print(i,j,y.shape,y_hat.shape,fmap_r[j].shape,fmap_g[j].shape)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
