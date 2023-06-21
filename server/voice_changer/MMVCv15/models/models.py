import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .modules import ResidualCouplingLayer, Flip, WN, ResBlock1, ResBlock2, LRELU_SLOPE

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .commons import init_weights, get_padding, sequence_mask
from .generator import SiFiGANGenerator
from .features import SignalGenerator, dilated_factor


class TextEncoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, requires_grad=True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        # パラメータを学習しない
        if requires_grad is False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, x_lengths):
        x = torch.transpose(x.half(), 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0, requires_grad=True):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

        # パラメータを学習しない
        if requires_grad is False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, requires_grad=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        # パラメータを学習しない
        if requires_grad is False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, requires_grad=True):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if requires_grad is False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, g=None):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        # periods = [2,3,5,7,11]
        periods = [3, 5, 7, 11, 13]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat, flag=True):
        if flag:
            y_d_rs = []
            y_d_gs = []
            fmap_rs = []
            fmap_gs = []
            for i, d in enumerate(self.discriminators):
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
                y_d_rs.append(y_d_r)
                y_d_gs.append(y_d_g)
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

            return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        else:
            y_d_gs = []
            with torch.no_grad():
                for i, d in enumerate(self.discriminators):
                    y_d_g, _ = d(y_hat)
                    y_d_gs.append(y_d_g)

            return y_d_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_flow,
        dec_out_channels=1,
        dec_kernel_size=7,
        n_speakers=0,
        gin_channels=0,
        requires_grad_pe=True,
        requires_grad_flow=True,
        requires_grad_text_enc=True,
        requires_grad_dec=True,
        requires_grad_emb_g=True,
        sample_rate=24000,
        hop_size=128,
        sine_amp=0.1,
        noise_amp=0.003,
        signal_types=["sine"],
        dense_factors=[0.5, 1, 4, 8],
        upsample_scales=[8, 4, 2, 2],
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.dec_out_channels = dec_out_channels
        self.dec_kernel_size = dec_kernel_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.requires_grad_pe = requires_grad_pe
        self.requires_grad_flow = requires_grad_flow
        self.requires_grad_text_enc = requires_grad_text_enc
        self.requires_grad_dec = requires_grad_dec
        self.requires_grad_emb_g = requires_grad_emb_g
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp
        self.signal_types = signal_types
        self.dense_factors = dense_factors
        self.upsample_scales = upsample_scales

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels, requires_grad=requires_grad_pe)
        self.enc_p = TextEncoder(inter_channels, hidden_channels, requires_grad=requires_grad_text_enc)
        self.dec = SiFiGANGenerator(in_channels=inter_channels, out_channels=dec_out_channels, channels=upsample_initial_channel, kernel_size=dec_kernel_size, upsample_scales=upsample_rates, upsample_kernel_sizes=upsample_kernel_sizes, requires_grad=requires_grad_dec)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, n_flows=n_flow, gin_channels=gin_channels, requires_grad=requires_grad_flow)
        self.signal_generator = SignalGenerator(sample_rate=sample_rate, hop_size=hop_size, noise_amp=noise_amp, signal_types=signal_types)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            self.emb_g.requires_grad = requires_grad_emb_g

    def forward(self, x, x_lengths, y, y_lengths, f0, slice_id, sid=None, target_ids=None):
        pass
        # sin, d = self.make_sin_d(f0)

        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        # # target sid 作成
        # target_sids = self.make_random_target_sids(target_ids, sid)

        # if self.n_speakers > 0:
        #     g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        #     tgt_g = self.emb_g(target_sids).unsqueeze(-1)  # [b, h, 1]
        # else:
        #     g = None

        # # PE
        # z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        # # Flow
        # z_p = self.flow(z, y_mask, g=g)
        # # VC
        # tgt_z = self.flow(z_p, y_mask, g=tgt_g, reverse=True)
        # # アライメントの作成
        # liner_alignment = F.one_hot(torch.arange(0, x.shape[2] + 2)).cuda()
        # liner_alignment = torch.stack([liner_alignment for _ in range(x.shape[0])], axis=0)
        # liner_alignment = F.interpolate(liner_alignment.float(), size=(z.shape[2]), mode="linear", align_corners=True)
        # liner_alignment = liner_alignment[:, 1:-1, :]
        # # TextEncとPEのshape合わせ
        # m_p = torch.matmul(m_p, liner_alignment)
        # logs_p = torch.matmul(logs_p, liner_alignment)

        # # slice
        # z_slice = slice_segments(z, slice_id, self.segment_size)
        # # targetのslice
        # tgt_z_slice = slice_segments(tgt_z, slice_id, self.segment_size)
        # # Dec
        # o = self.dec(sin, z_slice, d, sid=g)
        # tgt_o = self.dec(sin, tgt_z_slice, d, sid=tgt_g)

        # return (o, tgt_o), slice_id, x_mask, y_mask, ((z, z_p, m_p), logs_p, m_q, logs_q)

    def make_sin_d(self, f0):
        # f0 から sin と d を作成
        # f0 : [b, 1, t]
        # sin : [b, 1, t]
        # d : [4][b, 1, t]
        prod_upsample_scales = np.cumprod(self.upsample_scales)
        dfs_batch = []
        for df, us in zip(self.dense_factors, prod_upsample_scales):
            dilated_tensor = dilated_factor(f0, self.sample_rate, df)
            # result += [torch.repeat_interleave(dilated_tensor, us, dim=1)]
            result = [torch.stack([dilated_tensor for _ in range(us)], -1).reshape(dilated_tensor.shape[0], -1)]
            dfs_batch.append(torch.cat(result, dim=0).unsqueeze(1))
        in_batch = self.signal_generator(f0)

        return in_batch, dfs_batch

    def make_random_target_sids(self, target_ids, sid):
        # target_sids は target_ids をランダムで埋める
        target_sids = torch.zeros_like(sid)
        for i in range(len(target_sids)):
            source_id = sid[i]
            deleted_target_ids = target_ids[target_ids != source_id]  # source_id と target_id が同じにならないよう sid と同じものを削除
            if len(deleted_target_ids) >= 1:
                target_sids[i] = deleted_target_ids[torch.randint(len(deleted_target_ids), (1,))]
            else:
                # target_id 候補が無いときは仕方ないので sid を使う
                target_sids[i] = source_id
        return target_sids

    def voice_conversion(self, y, y_lengths, f0, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        sin, d = self.make_sin_d(f0)
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, _, _, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        # print("VC", sin.device, d[0].device, g_tgt.device)
        o_hat = self.dec(sin, z_hat * y_mask, d, sid=g_tgt)
        return o_hat[0]

    def voice_ra_pa_db(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        o_hat = self.dec(z * y_mask, g=g_tgt)
        return o_hat, y_mask, (z)

    def voice_ra_pa_da(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        # g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        o_hat = self.dec(z * y_mask, g=g_src)
        return o_hat, y_mask, (z)

    def voice_conversion_cycle(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        z_p_hat = self.flow(z_hat, y_mask, g=g_tgt)
        z_hat_hat = self.flow(z_p_hat, y_mask, g=g_src, reverse=True)
        o_hat = self.dec(z_hat_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

    def save_synthesizer(self, path):
        enc_q = self.enc_q.state_dict()
        dec = self.dec.state_dict()
        emb_g = self.emb_g.state_dict()
        torch.save({"enc_q": enc_q, "dec": dec, "emb_g": emb_g}, path)

    def load_synthesizer(self, path):
        dict = torch.load(path, map_location="cpu")
        enc_q = dict["enc_q"]
        dec = dict["dec"]
        emb_g = dict["emb_g"]
        self.enc_q.load_state_dict(enc_q)
        self.dec.load_state_dict(dec)
        self.emb_g.load_state_dict(emb_g)
