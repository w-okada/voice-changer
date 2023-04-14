import sys
import os
import argparse
from distutils.util import strtobool
import torch
from torch import nn
from onnxsim import simplify
import onnx

from infer_pack.models import TextEncoder256, GeneratorNSF, PosteriorEncoder, ResidualCouplingBlock


class SynthesizerTrnMs256NSFsid_ONNX(nn.Module):
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
        sr,
        **kwargs
    ):

        super().__init__()
        if (type(sr) == type("strr")):
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
        # self.hop_length = hop_length#
        self.spk_embed_dim = spk_embed_dim
        self.enc_p = TextEncoder256(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels, sr=sr, is_half=kwargs["is_half"]
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)
        print("gin_channels:", gin_channels, "self.spk_embed_dim:", self.spk_embed_dim)

    def forward(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


def export2onnx(input_model, output_model, output_model_simple, is_half):
    cpt = torch.load(input_model, map_location="cpu")
    if is_half:
        dev = torch.device("cuda", index=0)
    else:
        dev = torch.device("cpu")

    net_g_onnx = SynthesizerTrnMs256NSFsid_ONNX(*cpt["config"], is_half=is_half)
    try:
        net_g_onnx.eval().to(dev)
    except:
        is_half = False
        dev = torch.device("cpu")
        net_g_onnx.eval().to(dev)

    net_g_onnx.load_state_dict(cpt["weight"], strict=False)
    if is_half:
        print("!!!!!!!!!!!!!!!!!! half")
        net_g_onnx = net_g_onnx.half()
    else:
        print("!!!!!!!!!!!!!!!!!! full")
        net_g_onnx = net_g_onnx.float()

    if is_half:
        feats = torch.HalfTensor(1, 2192, 256).to(dev)
    else:
        feats = torch.FloatTensor(1, 2192, 256).to(dev)
    p_len = torch.LongTensor([2192]).to(dev)
    pitch = torch.zeros(1, 2192, dtype=torch.int64).to(dev)

    pitchf = torch.FloatTensor(1, 2192).to(dev)
    sid = torch.LongTensor([0]).to(dev)

    input_names = ["feats", "p_len", "pitch", "pitchf", "sid"]
    output_names = ["audio", ]

    torch.onnx.export(net_g_onnx,
                      (
                          feats,
                          p_len,
                          pitch,
                          pitchf,
                          sid,
                      ),
                      output_model,
                      dynamic_axes={
                          "feats": [1],
                          "pitch": [1],
                          "pitchf": [1],
                      },
                      do_constant_folding=False,
                      opset_version=17,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names)

    model_onnx2 = onnx.load(output_model)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, output_model_simple)
