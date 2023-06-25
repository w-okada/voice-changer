import math

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
from torchaudio.functional.functional import _hz_to_mel, _mel_to_hz

from . import commons, modules
from .commons import get_padding, init_weights
from .transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1

class HarmonicEmbedder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, gin_channels, num_head, num_harmonic=0, f0_min=50., f0_max=1100., device="cuda"):
        super(HarmonicEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head
        self.num_harmonic = num_harmonic

        f0_mel_min = np.log(1 + f0_min / 700)
        f0_mel_max = np.log(1 + f0_max * (1 + num_harmonic) / 700)
        self.sequence = torch.from_numpy(np.linspace(f0_mel_min, f0_mel_max, num_embeddings-2))
        self.emb_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.linear_q = Conv1d(gin_channels, num_head * (1 + num_harmonic), 1)
        self.weight = None

    def forward(self, x, g):
        b, l = x.size()
        non_zero = (x != 0.).to(dtype=torch.long).unsqueeze(1)
        mel = torch.log(1 + x / 700).unsqueeze(1)
        harmonies = torch.arange(1 + self.num_harmonic, device=x.device, dtype=x.dtype).view(1, 1 + self.num_harmonic, 1) + 1.
        ix = torch.searchsorted(self.sequence.to(x.device), mel * harmonies).to(x.device) + 1
        ix = ix * non_zero
        emb = self.emb_layer(ix).transpose(1, 3).reshape(b, self.num_head, self.embedding_dim // self.num_head, 1 + self.num_harmonic, l)
        if self.weight is None:
            weight = torch.nn.functional.softmax(self.linear_q(g).reshape(b, self.num_head, 1, 1 + self.num_harmonic, 1), 3)
        else:
            weight = self.weight
        res = torch.sum(emb * weight, dim=3).reshape(b, self.embedding_dim, l)
        return res

    def fix_speaker(self, g):
        self.weight = torch.nn.functional.softmax(self.linear_q(g).reshape(1, self.num_head, 1, 1 + self.num_harmonic, 1), 3)

    def unfix_speaker(self, g):
        self.weight = None

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, dilation=1, bias=True):
        super(DilatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, dilation=dilation, bias=bias))

    def forward(self, x):
        x = torch.flip(x, [2])
        x = F.pad(x, [0, (self.kernel_size - 1) * self.dilation], mode="constant", value=0.)
        size = x.shape[2] // self.stride
        x = self.conv(x)[:, :, :size]
        x = torch.flip(x, [2])
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)


class CausalConvTranspose1d(nn.Module):
    """
    padding = 0, dilation = 1のとき

    Lout = (Lin - 1) * stride + kernel_rate * stride + output_padding
    Lout = Lin * stride + (kernel_rate - 1) * stride + output_padding
    output_paddingいらないね
    """
    def __init__(self, in_channels, out_channels, kernel_rate=3, stride=1, groups=1):
        super(CausalConvTranspose1d, self).__init__()
        kernel_size = kernel_rate * stride
        self.trim_size = (kernel_rate - 1) * stride
        self.conv = weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups))

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.trim_size]

    def remove_weight_norm(self):
        remove_weight_norm(self.conv)


class LoRALinear1d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.r = r
        self.main_fc = weight_norm(nn.Conv1d(in_channels, out_channels, 1))
        self.adapter_in = nn.Conv1d(info_channels, in_channels * r, 1)
        self.adapter_out = nn.Conv1d(info_channels, out_channels * r, 1)
        nn.init.normal_(self.adapter_in.weight.data, 0, 0.01)
        nn.init.constant_(self.adapter_out.weight.data, 1e-6)
        self.adapter_in = weight_norm(self.adapter_in)
        self.adapter_out = weight_norm(self.adapter_out)
        self.speaker_fixed = False

    def forward(self, x, g):
        x_ = self.main_fc(x)
        if not self.speaker_fixed:
            a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
            a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
            l = torch.einsum("brl,brc->bcl", torch.einsum("bcl,bcr->brl", x, a_in), a_out)
            x_ = x_ + l
        return x_

    def remove_weight_norm(self):
        remove_weight_norm(self.main_fc)
        remove_weight_norm(self.adapter_in)
        remove_weight_norm(self.adapter_out)

    def fix_speaker(self, g):
        self.speaker_fixed = True
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2)
        self.main_fc.weight.data.add_(weight)

    def unfix_speaker(self, g):
        self.speaker_fixed = False
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2)
        self.main_fc.weight.data.sub_(weight)


class LoRALinear2d(nn.Module):
    def __init__(self, in_channels, out_channels, info_channels, r):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.info_channels = info_channels
        self.r = r
        self.main_fc = weight_norm(nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1)))
        self.adapter_in = nn.Conv1d(info_channels, in_channels * r, 1)
        self.adapter_out = nn.Conv1d(info_channels, out_channels * r, 1)
        nn.init.normal_(self.adapter_in.weight.data, 0, 0.01)
        nn.init.constant_(self.adapter_out.weight.data, 1e-6)
        self.adapter_in = weight_norm(self.adapter_in)
        self.adapter_out = weight_norm(self.adapter_out)
        self.speaker_fixed = False

    def forward(self, x, g):
        x_ = self.main_fc(x)
        if not self.speaker_fixed:
            a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
            a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
            l = torch.einsum("brhw,brc->bchw", torch.einsum("bchw,bcr->brhw", x, a_in), a_out)
            x_ = x_ + l
        return x_

    def remove_weight_norm(self):
        remove_weight_norm(self.main_fc)
        remove_weight_norm(self.adapter_in)
        remove_weight_norm(self.adapter_out)

    def fix_speaker(self, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2).unsqueeze(3)
        self.main_fc.weight.data.add_(weight)

    def unfix_speaker(self, g):
        a_in = self.adapter_in(g).view(-1, self.in_channels, self.r)
        a_out = self.adapter_out(g).view(-1, self.r, self.out_channels)
        weight = torch.einsum("bir,bro->oi", a_in, a_out).unsqueeze(2).unsqueeze(3)
        self.main_fc.weight.data.sub_(weight)


class MBConv2d(torch.nn.Module):
    """
    Causal MBConv2D
    """
    def __init__(self, in_channels, out_channels, gin_channels, kernel_size, stride, extend_ratio, r, use_spectral_norm=False):
        super(MBConv2d, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.kernel_size = kernel_size
        self.pwconv1 = LoRALinear2d(in_channels, inner_channels, gin_channels, r=r)
        self.dwconv = norm_f(Conv2d(inner_channels, inner_channels, kernel_size, stride, groups=inner_channels))
        self.pwconv2 = LoRALinear2d(inner_channels, out_channels, gin_channels, r=r)
        self.pwnorm = LayerNorm(in_channels)
        self.dwnorm = LayerNorm(inner_channels)

    def forward(self, x, g):
        x = self.pwnorm(x)
        x = self.pwconv1(x, g)
        x = F.pad(x, [0, 0, self.kernel_size[0] - 1, 0], mode="constant")
        x = self.dwnorm(x)
        x = self.dwconv(x)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = self.pwconv2(x, g)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        return x

class ConvNext2d(torch.nn.Module):
    """
    Causal ConvNext Block
    stride = 1 only
    """
    def __init__(self, in_channels, out_channels, gin_channels, kernel_size, stride, extend_ratio, r, use_spectral_norm=False):
        super(ConvNext2d, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        inner_channels = int(in_channels * extend_ratio)
        self.kernel_size = kernel_size
        self.dwconv = norm_f(Conv2d(in_channels, in_channels, kernel_size, stride, groups=in_channels))
        self.pwconv1 = LoRALinear2d(in_channels, inner_channels, gin_channels, r=r)
        self.pwconv2 = LoRALinear2d(inner_channels, out_channels, gin_channels, r=r)
        self.act = nn.GELU()
        self.norm = LayerNorm(in_channels)

    def forward(self, x, g):
        x = F.pad(x, [0, 0, self.kernel_size[0] - 1, 0], mode="constant")
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x, g)
        x = self.act(x)
        x = self.pwconv2(x, g)
        x = self.act(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.dwconv)


class WaveBlock(torch.nn.Module):
    def __init__(self, inner_channels, gin_channels, kernel_sizes, strides, dilations, extend_rate, r):
        super(WaveBlock, self).__init__()
        norm_f = weight_norm
        extend_channels = int(inner_channels * extend_rate)
        self.dconvs = nn.ModuleList()
        self.p1convs = nn.ModuleList()
        self.p2convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()

        # self.ses = nn.ModuleList()
        # self.norms = []
        for i, (k, s, d) in enumerate(zip(kernel_sizes, strides, dilations)):
            self.dconvs.append(DilatedCausalConv1d(inner_channels, inner_channels, k, stride=s, dilation=d, groups=inner_channels))
            self.p1convs.append(LoRALinear1d(inner_channels, extend_channels, gin_channels, r))
            self.p2convs.append(LoRALinear1d(extend_channels, inner_channels, gin_channels, r))
            self.norms.append(LayerNorm(inner_channels))

    def forward(self, x, x_mask, g):
        x *= x_mask
        for i in range(len(self.dconvs)):
            residual = x.clone()
            x = self.dconvs[i](x)
            x = self.norms[i](x)
            x *= x_mask
            x = self.p1convs[i](x, g)
            x = self.act(x)
            x = self.p2convs[i](x, g)
            x = residual + x
        return x

    def remove_weight_norm(self):
        for c in self.dconvs:
            c.remove_weight_norm()
        for c in self.p1convs:
            c.remove_weight_norm()
        for c in self.p2convs:
            c.remove_weight_norm()

    def fix_speaker(self, g):
        for c in self.p1convs:
            c.fix_speaker(g)
        for c in self.p2convs:
            c.fix_speaker(g)

    def unfix_speaker(self, g):
        for c in self.p1convs:
            c.unfix_speaker(g)
        for c in self.p2convs:
            c.unfix_speaker(g)


class SnakeFilter(torch.nn.Module):
    """
    Adaptive filter using snakebeta
    """
    def __init__(self, channels, groups, kernel_size, num_layers, eps=1e-6):
        super(SnakeFilter, self).__init__()
        self.eps = eps
        self.num_layers = num_layers
        inner_channels = channels * groups
        self.init_conv = DilatedCausalConv1d(1, inner_channels, kernel_size)
        self.dconvs = torch.nn.ModuleList()
        self.pconvs = torch.nn.ModuleList()
        self.post_conv = DilatedCausalConv1d(inner_channels+1, 1, kernel_size, bias=False)

        for i in range(self.num_layers):
            self.dconvs.append(DilatedCausalConv1d(inner_channels, inner_channels, kernel_size, stride=1, groups=inner_channels, dilation=kernel_size ** (i + 1)))
            self.pconvs.append(weight_norm(Conv1d(inner_channels, inner_channels, 1, groups=groups)))
        self.snake_alpha = torch.nn.Parameter(torch.zeros(inner_channels), requires_grad=True)
        self.snake_beta = torch.nn.Parameter(torch.zeros(inner_channels), requires_grad=True)

    def forward(self, x):
        y = x.clone()
        x = self.init_conv(x)
        for i in range(self.num_layers):
            # snake activation
            x = self.dconvs[i](x)
            x = self.pconvs[i](x)
        x = x + (1.0 / torch.clip(self.snake_beta.unsqueeze(0).unsqueeze(-1), min=self.eps)) * torch.pow(torch.sin(x * self.snake_alpha.unsqueeze(0).unsqueeze(-1)), 2)
        x = torch.cat([x, y], 1)
        x = self.post_conv(x)
        return x

    def remove_weight_norm(self):
        self.init_conv.remove_weight_norm()
        for c in self.dconvs:
            c.remove_weight_norm()
        for c in self.pconvs:
            remove_weight_norm(c)
        self.post_conv.remove_weight_norm()

"""
https://github.com/charactr-platform/vocos/blob/main/vocos/heads.py
"""
class FourierHead(nn.Module):
    """Base class for inverse fourier modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len * 2
        N = frame_len
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(N * 2)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(1j * torch.pi * n0 * torch.arange(N * 2) / N)
        post_twiddle = torch.exp(1j * torch.pi * (torch.arange(N * 2) + n0) / (N * 2))
        self.register_buffer("pre_twiddle", torch.view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", torch.view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, N, L), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        X = X.transpose(1, 2)
        B, L, N = X.shape
        Y = torch.zeros((B, L, N * 2), dtype=X.dtype, device=X.device)
        Y[..., :N] = X
        Y[..., N:] = -1 * torch.conj(torch.flip(X, dims=(-1,)))
        y = torch.fft.ifft(Y * torch.view_as_complex(self.pre_twiddle).expand(Y.shape), dim=-1)
        y = torch.real(y * torch.view_as_complex(self.post_twiddle).expand(y.shape)) * np.sqrt(N) * np.sqrt(2)
        result = y * self.window.expand(y.shape)
        output_size = (1, (L + 1) * N)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        elif self.padding == "same":
            pad = self.frame_len // 4
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        audio = audio[:, pad:-pad]
        return audio.unsqueeze(1)


class IMDCTSymExpHead(FourierHead):
    """
    IMDCT Head module for predicting MDCT coefficients with symmetric exponential function

    Args:
        dim (int): Hidden dimension of the model.
        mdct_frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
        sample_rate (int, optional): The sample rate of the audio. If provided, the last layer will be initialized
                                     based on perceptual scaling. Defaults to None.
        clip_audio (bool, optional): Whether to clip the audio output within the range of [-1.0, 1.0]. Defaults to False.
    """

    def __init__(
        self, dim: int, gin_channels: int, mdct_frame_len: int, padding: str = "same", sample_rate: int = 24000,
    ):
        super().__init__()
        out_dim = mdct_frame_len
        self.dconv = DilatedCausalConv1d(dim, dim, 5, 1, dim, 1)
        self.pconv1 = LoRALinear1d(dim, dim * 2, gin_channels, 2)
        self.pconv2 = LoRALinear1d(dim * 2, out_dim, gin_channels, 2)
        self.act = torch.nn.GELU()
        self.imdct = IMDCT(frame_len=mdct_frame_len, padding=padding)

        if sample_rate is not None:
            # optionally init the last layer following mel-scale
            m_max = _hz_to_mel(sample_rate // 2)
            m_pts = torch.linspace(0, m_max, out_dim)
            f_pts = _mel_to_hz(m_pts)
            scale = 1 - (f_pts / f_pts.max())

            with torch.no_grad():
                self.pconv2.main_fc.weight.mul_(scale.view(-1, 1, 1))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the IMDCTSymExpHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.dconv(x)
        x = self.pconv1(x, g)
        x = self.act(x)
        x = self.pconv2(x, g)
        x = symexp(x)
        x = torch.clip(x, min=-1e2, max=1e2)  # safeguard to prevent excessively large magnitudes
        audio = self.imdct(x)
        return audio

    def remove_weight_norm(self):
        self.dconv.remove_weight_norm()
        self.pconv1.remove_weight_norm()
        self.pconv2.remove_weight_norm()

    def fix_speaker(self, g):
        self.pconv1.fix_speaker(g)
        self.pconv2.fix_speaker(g)

    def unfix_speaker(self, g):
        self.pconv1.unfix_speaker(g)
        self.pconv2.unfix_speaker(g)

def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)