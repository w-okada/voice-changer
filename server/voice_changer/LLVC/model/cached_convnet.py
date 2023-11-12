# based on https://github.com/YangangCao/Causal-U-Net/blob/main/cunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Based on https://github.com/f90/Seq-U-Net/blob/master/sequnet_res.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, use_2d):
        super().__init__()
        self.use_2d = use_2d
        if use_2d:
            self.filter = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.filter = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            self.dropout = nn.Dropout1d(dropout)
        self.output_crop = dilation * (kernel_size - 1)

    def forward(self, x):
        filtered = torch.tanh(self.filter(x))
        gated = torch.sigmoid(self.gate(x))
        residual = filtered * gated
        # pad dim 1 of x to match residual
        if self.use_2d:
            x = F.pad(x, (0, 0, 0, 0, 0, residual.shape[1] - x.shape[1]))
            output = x[..., self.output_crop :, self.output_crop :] + residual
        else:
            x = F.pad(x, (0, 0, 0, residual.shape[1] - x.shape[1]))
            output = x[..., self.output_crop :] + residual
        output = self.dropout(output)
        return output


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, use_2d):
        super().__init__()
        if use_2d:
            conv_layer = nn.Conv2d
            batchnorm_layer = nn.BatchNorm2d
            dropout_layer = nn.Dropout2d
        else:
            conv_layer = nn.Conv1d
            batchnorm_layer = nn.BatchNorm1d
            dropout_layer = nn.Dropout1d
        self.conv = nn.Sequential(
            conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation),
            batchnorm_layer(num_features=out_channels),
            dropout_layer(dropout),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        """
        1D Causal convolution.
        """
        return self.conv(x)


class CachedConvNet(nn.Module):
    def __init__(self, num_channels, kernel_sizes, dilations, dropout, combine_residuals, use_residual_blocks, out_channels, use_2d, use_pool=False, pool_kernel=2):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), "kernel_sizes and dilations must be the same length"
        assert len(kernel_sizes) == len(out_channels), "kernel_sizes and out_channels must be the same length"
        self.num_layers = len(kernel_sizes)
        self.ctx_height = max(out_channels)
        self.down_convs = nn.ModuleList()
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.combine_residuals = combine_residuals
        self.use_2d = use_2d
        self.use_pool = use_pool

        # compute buffer lengths for each layer
        self.buf_lengths = [(k - 1) * d for k, d in zip(kernel_sizes, dilations)]

        # Compute buffer start indices for each layer
        self.buf_indices = [0]
        for i in range(len(kernel_sizes) - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        if use_residual_blocks:
            block = ResidualBlock
        else:
            block = CausalConvBlock

        if self.use_pool:
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel)

        for i in range(self.num_layers):
            in_channel = num_channels if i == 0 else out_channels[i - 1]
            self.down_convs.append(block(in_channels=in_channel, out_channels=out_channels[i], kernel_size=kernel_sizes[i], dilation=dilations[i], dropout=dropout, use_2d=use_2d))

    def init_ctx_buf(self, batch_size, device, height=None):
        """
        Initialize context buffer for each layer.
        """
        if height is not None:
            up_ctx = torch.zeros((batch_size, self.ctx_height, height, sum(self.buf_lengths))).to(device)
        else:
            up_ctx = torch.zeros((batch_size, self.ctx_height, sum(self.buf_lengths))).to(device)
        return up_ctx

    def forward(self, x, ctx):
        """
        Args:
            x: [B, in_channels, T]
                Input
            ctx: {[B, channels, self.buf_length[0]], ...}
                A list of tensors holding context for each unet layer. (len(ctx) == self.num_layers)
        Returns:
            x: [B, out_channels, T]
            ctx: {[B, channels, self.buf_length[0]], ...}
                Updated context buffer with output as the
                last element.
        """
        if self.use_pool:
            x = self.pool(x)

        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            # concatenate context buffer with input
            if self.use_2d:
                conv_in = torch.cat((ctx[..., : x.shape[1], : x.shape[-2], buf_start_idx:buf_end_idx], x), dim=-1)
            else:
                conv_in = torch.cat((ctx[..., : x.shape[-2], buf_start_idx:buf_end_idx], x), dim=-1)

            # Push current output to the context buffer
            if self.use_2d:
                ctx[..., : x.shape[1], : x.shape[-2], buf_start_idx:buf_end_idx] = conv_in[..., -self.buf_lengths[i] :]
            else:
                ctx[..., : x.shape[1], buf_start_idx:buf_end_idx] = conv_in[..., -self.buf_lengths[i] :]

            # pad second-to-last index of input with self.buf_lengths[i] // 2 zeros
            # on each side to ensure that height of output is the same as input
            if self.use_2d:
                conv_in = F.pad(conv_in, (0, 0, self.buf_lengths[i] // 2, self.buf_lengths[i] // 2))

            if self.combine_residuals == "add":
                x = x + self.down_convs[i](conv_in)
            elif self.combine_residuals == "multiply":
                x = x * self.down_convs[i](conv_in)
            else:
                x = self.down_convs[i](conv_in)

        if self.use_pool:
            x = F.interpolate(x, scale_factor=self.pool.kernel_size[0])

        return x, ctx
