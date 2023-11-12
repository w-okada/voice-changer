import math
from collections import OrderedDict
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from voice_changer.LLVC.model.cached_convnet import CachedConvNet


class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).
    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size))

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


def mod_pad(x, chunk_size, pad):
    # Mod pad the input to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod


class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        """
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = super().forward(x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation),
            LayerNormPermuted(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            LayerNormPermuted(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DilatedCausalConvEncoder(nn.Module):
    """
    A dilated causal convolution based encoder for encoding
    time domain audio input into latent space.
    """

    def __init__(self, channels, num_layers, kernel_size=3):
        super(DilatedCausalConvEncoder, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Compute buffer lengths for each layer
        # buf_length[i] = (kernel_size - 1) * dilation[i]
        self.buf_lengths = [(kernel_size - 1) * 2**i for i in range(num_layers)]

        # Compute buffer start indices for each layer
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        # Dilated causal conv layers aggregate previous context to obtain
        # contexful encoded input.
        _dcc_layers = OrderedDict()
        for i in range(num_layers):
            dcc_layer = DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, padding=0, dilation=2**i)
            _dcc_layers.update({"dcc_%d" % i: dcc_layer})
        self.dcc_layers = nn.Sequential(_dcc_layers)

    def init_ctx_buf(self, batch_size, device):
        """
        Returns an initialized context buffer for a given batch size.
        """
        return torch.zeros((batch_size, self.channels, (self.kernel_size - 1) * (2**self.num_layers - 1)), device=device)

    def forward(self, x, ctx_buf):
        """
        Encodes input audio `x` into latent space, and aggregates
        contextual information in `ctx_buf`. Also generates new context
        buffer with updated context.
        Args:
            x: [B, in_channels, T]
                Input multi-channel audio.
            ctx_buf: {[B, channels, self.buf_length[0]], ...}
                A list of tensors holding context for each dilation
                causal conv layer. (len(ctx_buf) == self.num_layers)
        Returns:
            ctx_buf: {[B, channels, self.buf_length[0]], ...}
                Updated context buffer with output as the
                last element.
        """
        T = x.shape[-1]  # Sequence length # noqa

        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            # DCC input: concatenation of current output and context
            dcc_in = torch.cat((ctx_buf[..., buf_start_idx:buf_end_idx], x), dim=-1)

            # Push current output to the context buffer
            ctx_buf[..., buf_start_idx:buf_end_idx] = dcc_in[..., -self.buf_lengths[i] :]

            # Residual connection
            x = x + self.dcc_layers[i](dcc_in)

        return x, ctx_buf


class CausalTransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
    """
    Adapted from:
    "https://github.com/alexmt-scale/causal-transformer-decoder/blob/"
    "0caf6ad71c46488f76d89845b0123d2550ef792f/"
    "causal_transformer_decoder/model.py#L77"
    """

    def forward2(self, tgt: Tensor, memory: Optional[Tensor] = None, chunk_size: int = 1):
        tgt_last_tok = tgt[:, -chunk_size:, :]

        # self attention part
        tmp_tgt, sa_map = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=None,
        )
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        if memory is not None:
            tmp_tgt, ca_map = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=None,  # Attend to the entire chunk
                key_padding_mask=None,
            )
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt_last_tok))))
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok, sa_map, ca_map


class CausalTransformerDecoder(nn.Module):
    """
    A casual transformer decoder which decodes input vectors using
    precisely `ctx_len` past vectors in the sequence, and using no future
    vectors at all.
    """

    def __init__(self, model_dim, ctx_len, chunk_size, num_layers, nhead, use_pos_enc, ff_dim, dropout):
        super(CausalTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.unfold = nn.Unfold(kernel_size=(ctx_len + chunk_size, 1), stride=chunk_size)
        self.pos_enc = PositionalEncoding(model_dim, max_len=200)
        self.tf_dec_layers = nn.ModuleList([CausalTransformerDecoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim, batch_first=True, dropout=dropout) for _ in range(num_layers)])

    def init_ctx_buf(self, batch_size, device):
        return torch.zeros((batch_size, self.num_layers + 1, self.ctx_len, self.model_dim), device=device)

    def _causal_unfold(self, x):
        """
        Unfolds the sequence into a batch of sequences
        prepended with `ctx_len` previous values.

        Args:
            x: [B, ctx_len + L, C]
            ctx_len: int
        Returns:
            [B * L, ctx_len + 1, C]
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, ctx_len + L]
        x = self.unfold(x.unsqueeze(-1))  # [B, C * (ctx_len + chunk_size), -1]
        x = x.permute(0, 2, 1)
        x = x.reshape(B, -1, C, self.ctx_len + self.chunk_size)
        x = x.reshape(-1, C, self.ctx_len + self.chunk_size)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, tgt, mem, ctx_buf, probe=False):
        """
        Args:
            x: [B, model_dim, T]
            ctx_buf: [B, num_layers, model_dim, ctx_len]
        """
        mem, _ = mod_pad(mem, self.chunk_size, (0, 0))
        tgt, mod = mod_pad(tgt, self.chunk_size, (0, 0))

        # Input sequence length
        B, C, T = tgt.shape

        tgt = tgt.permute(0, 2, 1)
        mem = mem.permute(0, 2, 1)

        # Prepend mem with the context
        mem = torch.cat((ctx_buf[:, 0, :, :], mem), dim=1)
        ctx_buf[:, 0, :, :] = mem[:, -self.ctx_len :, :]
        mem_ctx = self._causal_unfold(mem)
        if self.use_pos_enc:
            mem_ctx = mem_ctx + self.pos_enc(mem_ctx)

        # Attention chunk size: required to ensure the model
        # wouldn't trigger an out-of-memory error when working
        # on long sequences.
        K = 1000

        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            # Update the tgt with context
            tgt = torch.cat((ctx_buf[:, i + 1, :, :], tgt), dim=1)
            ctx_buf[:, i + 1, :, :] = tgt[:, -self.ctx_len :, :]

            # Compute encoded output
            tgt_ctx = self._causal_unfold(tgt)
            if self.use_pos_enc and i == 0:
                tgt_ctx = tgt_ctx + self.pos_enc(tgt_ctx)
            tgt = torch.zeros_like(tgt_ctx)[:, -self.chunk_size :, :]
            for i in range(int(math.ceil(tgt.shape[0] / K))):
                tgt[i * K : (i + 1) * K], _sa_map, _ca_map = tf_dec_layer.forward2(tgt_ctx[i * K : (i + 1) * K], mem_ctx[i * K : (i + 1) * K], self.chunk_size)
            tgt = tgt.reshape(B, T, C)

        tgt = tgt.permute(0, 2, 1)
        if mod != 0:
            tgt = tgt[..., :-mod]

        return tgt, ctx_buf


class MaskNet(nn.Module):
    def __init__(self, enc_dim, num_enc_layers, dec_dim, dec_buf_len, dec_chunk_size, num_dec_layers, use_pos_enc, skip_connection, proj, decoder_dropout):
        super(MaskNet, self).__init__()
        self.skip_connection = skip_connection
        self.proj = proj

        # Encoder based on dilated causal convolutions.
        self.encoder = DilatedCausalConvEncoder(channels=enc_dim, num_layers=num_enc_layers)

        # Project between encoder and decoder dimensions
        self.proj_e2d_e = nn.Sequential(nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim), nn.ReLU())
        self.proj_e2d_l = nn.Sequential(nn.Conv1d(enc_dim, dec_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim), nn.ReLU())
        self.proj_d2e = nn.Sequential(nn.Conv1d(dec_dim, enc_dim, kernel_size=1, stride=1, padding=0, groups=dec_dim), nn.ReLU())

        # Transformer decoder that operates on chunks of size
        # buffer size.

        self.decoder = CausalTransformerDecoder(model_dim=dec_dim, ctx_len=dec_buf_len, chunk_size=dec_chunk_size, num_layers=num_dec_layers, nhead=8, use_pos_enc=use_pos_enc, ff_dim=2 * dec_dim, dropout=decoder_dropout)

    def forward(self, x, l, enc_buf, dec_buf):  # noqa
        """
        Generates a mask based on encoded input `e` and the one-hot
        label `label`.

        Args:
            x: [B, C, T]
                Input audio sequence
            l: [B, C]
                Label embedding
            ctx_buf: {[B, C, <receptive field of the layer>], ...}
                List of context buffers maintained by DCC encoder
        """
        # Enocder the label integrated input
        e, enc_buf = self.encoder(x, enc_buf)

        # Label integration
        l = l.unsqueeze(2) * e  # noqa

        # Project to `dec_dim` dimensions
        if self.proj:
            e = self.proj_e2d_e(e)
            m = self.proj_e2d_l(l)
            # Cross-attention to predict the mask
            m, dec_buf = self.decoder(m, e, dec_buf)
        else:
            # Cross-attention to predict the mask
            m, dec_buf = self.decoder(l, e, dec_buf)

        # Project mask to encoder dimensions
        if self.proj:
            m = self.proj_d2e(m)

        # Final mask after residual connection
        if self.skip_connection:
            m = l + m

        return m, enc_buf, dec_buf


class Net(nn.Module):
    def __init__(self, label_len, L=8, enc_dim=512, num_enc_layers=10, dec_dim=256, dec_buf_len=100, num_dec_layers=2, dec_chunk_size=72, out_buf_len=2, use_pos_enc=True, skip_connection=True, proj=True, lookahead=True, decoder_dropout=0.0, convnet_config=None):
        super(Net, self).__init__()
        self.L = L
        self.dec_chunk_size = dec_chunk_size
        self.out_buf_len = out_buf_len
        self.enc_dim = enc_dim
        self.lookahead = lookahead

        self.convnet_config = convnet_config
        if convnet_config["convnet_prenet"]:
            self.convnet_pre = CachedConvNet(1, convnet_config["kernel_sizes"], convnet_config["dilations"], convnet_config["dropout"], convnet_config["combine_residuals"], convnet_config["use_residual_blocks"], convnet_config["out_channels"], use_2d=False)

        # Input conv to convert input audio to a latent representation
        kernel_size = 3 * L if lookahead else L
        self.in_conv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=enc_dim, kernel_size=kernel_size, stride=L, padding=0, bias=False), nn.ReLU())

        # Label embedding layer
        label_len = 1
        self.label_embedding = nn.Sequential(nn.Linear(label_len, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, enc_dim), nn.LayerNorm(enc_dim), nn.ReLU())

        # Mask generator
        self.mask_gen = MaskNet(enc_dim=enc_dim, num_enc_layers=num_enc_layers, dec_dim=dec_dim, dec_buf_len=dec_buf_len, dec_chunk_size=dec_chunk_size, num_dec_layers=num_dec_layers, use_pos_enc=use_pos_enc, skip_connection=skip_connection, proj=proj, decoder_dropout=decoder_dropout)

        # Output conv layer
        self.out_conv = nn.Sequential(nn.ConvTranspose1d(in_channels=enc_dim, out_channels=1, kernel_size=(out_buf_len + 1) * L, stride=L, padding=out_buf_len * L, bias=False), nn.Tanh())

    def init_buffers(self, batch_size, device):
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size, device)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size, device)
        out_buf = torch.zeros(batch_size, self.enc_dim, self.out_buf_len, device=device)
        return enc_buf, dec_buf, out_buf

    def forward(self, x, init_enc_buf=None, init_dec_buf=None, init_out_buf=None, convnet_pre_ctx=None, pad=True):
        """
        Extracts the audio corresponding to the `label` in the given
        `mixture`. Generates `chunk_size` samples per iteration.

        Args:
            mixed: [B, n_mics, T]
                input audio mixture
            label: [B, num_labels]
                one hot label
        Returns:
            out: [B, n_spk, T]
                extracted audio with sounds corresponding to the `label`
        """
        label = torch.zeros(x.shape[0], 1, device=x.device)
        mod = 0
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.L, pad=pad_size)

        if hasattr(self, "convnet_pre"):
            if convnet_pre_ctx is None:
                convnet_pre_ctx = self.convnet_pre.init_ctx_buf(x.shape[0], x.device)

            convnet_out, convnet_pre_ctx = self.convnet_pre(x, convnet_pre_ctx)

            if self.convnet_config["skip_connection"] == "add":
                x = x + convnet_out
            elif self.convnet_config["skip_connection"] == "multiply":
                x = x * convnet_out
            else:
                x = convnet_out

        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            assert init_enc_buf is None and init_dec_buf is None and init_out_buf is None, "Both buffers have to initialized, or " "both of them have to be None."
            enc_buf, dec_buf, out_buf = self.init_buffers(x.shape[0], x.device)
        else:
            (
                enc_buf,
                dec_buf,
                out_buf,
            ) = (
                init_enc_buf,
                init_dec_buf,
                init_out_buf,
            )

        # Generate latent space representation of the input
        x = self.in_conv(x)

        # Generate label embedding
        l = self.label_embedding(label)  # [B, label_len] --> [B, channels]  # noqa

        # Generate mask corresponding to the label
        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)

        # Apply mask and decode
        x = x * m
        x = torch.cat((out_buf, x), dim=-1)
        out_buf = x[..., -self.out_buf_len :]
        x = self.out_conv(x)

        # Remove mod padding, if present.
        if mod != 0:
            x = x[:, :, :-mod]

        if init_enc_buf is None:
            return x
        else:
            return x, enc_buf, dec_buf, out_buf, convnet_pre_ctx
