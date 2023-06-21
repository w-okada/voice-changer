# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""HiFiGAN and SiFiGAN Generator modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG
    - https://github.com/jik876/hifi-gan

"""

from logging import getLogger

import torch.nn as nn
from .residual_block import AdaptiveResidualBlock, Conv1d, ResidualBlock

# A logger for this file
logger = getLogger(__name__)


class SiFiGANGenerator(nn.Module):
    """SiFiGAN generator module."""

    def __init__(
        self,
        in_channels,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(5, 4, 3, 2),
        upsample_kernel_sizes=(10, 8, 6, 4),
        source_network_params={
            "resblock_kernel_size": 3,  # currently only 3 is supported.
            "resblock_dilations": [(1,), (1, 2), (1, 2, 4), (1, 2, 4, 8)],
            "use_additional_convs": True,
        },
        filter_network_params={
            "resblock_kernel_sizes": (3, 5, 7),
            "resblock_dilations": [(1, 3, 5), (1, 3, 5), (1, 3, 5)],
            "use_additional_convs": False,
        },
        share_upsamples=False,
        share_downsamples=False,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        requires_grad=True,
    ):
        """Initialize SiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            source_network_params (dict): Parameters for source-network.
            filter_network_params (dict): Parameters for filter-network.
            share_upsamples (bool): Whether to share up-sampling transposed CNNs.
            share_downsamples (bool): Whether to share down-sampling CNNs.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.source_network_params = source_network_params
        self.filter_network_params = filter_network_params
        self.share_upsamples = share_upsamples
        self.share_downsamples = share_downsamples
        self.sn = nn.ModuleDict()
        self.fn = nn.ModuleDict()
        self.input_conv = Conv1d(
            in_channels,
            channels,
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        self.sn["upsamples"] = nn.ModuleList()
        self.fn["upsamples"] = nn.ModuleList()
        self.sn["blocks"] = nn.ModuleList()
        self.fn["blocks"] = nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.sn["upsamples"] += [
                nn.Sequential(
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                        bias=bias,
                    ),
                )
            ]
            if not share_upsamples:
                self.fn["upsamples"] += [
                    nn.Sequential(
                        getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                        nn.ConvTranspose1d(
                            channels // (2**i),
                            channels // (2 ** (i + 1)),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                            output_padding=upsample_scales[i] % 2,
                            bias=bias,
                        ),
                    )
                ]
            self.sn["blocks"] += [
                AdaptiveResidualBlock(
                    kernel_size=source_network_params["resblock_kernel_size"],
                    channels=channels // (2 ** (i + 1)),
                    dilations=source_network_params["resblock_dilations"][i],
                    bias=bias,
                    use_additional_convs=source_network_params["use_additional_convs"],
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]
            for j in range(len(filter_network_params["resblock_kernel_sizes"])):
                self.fn["blocks"] += [
                    ResidualBlock(
                        kernel_size=filter_network_params["resblock_kernel_sizes"][j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=filter_network_params["resblock_dilations"][j],
                        bias=bias,
                        use_additional_convs=filter_network_params["use_additional_convs"],
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.sn["output_conv"] = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.fn["output_conv"] = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            nn.Tanh(),
        )

        # sine embedding layers
        self.sn["emb"] = Conv1d(
            1,
            channels // (2 ** len(upsample_kernel_sizes)),
            kernel_size,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        # down-sampling CNNs
        self.sn["downsamples"] = nn.ModuleList()
        for i in reversed(range(1, len(upsample_kernel_sizes))):
            self.sn["downsamples"] += [
                nn.Sequential(
                    nn.Conv1d(
                        channels // (2 ** (i + 1)),
                        channels // (2**i),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] - (upsample_kernel_sizes[i] % 2 == 0),
                        bias=bias,
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
        if not share_downsamples:
            self.fn["downsamples"] = nn.ModuleList()
            for i in reversed(range(1, len(upsample_kernel_sizes))):
                self.fn["downsamples"] += [
                    nn.Sequential(
                        nn.Conv1d(
                            channels // (2 ** (i + 1)),
                            channels // (2**i),
                            upsample_kernel_sizes[i],
                            upsample_scales[i],
                            padding=upsample_scales[i] - (upsample_kernel_sizes[i] % 2 == 0),
                            bias=bias,
                        ),
                        getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                    )
                ]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        if requires_grad is False:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, c, d, sid):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input sine signal (B, 1, T).
            c (Tensor): Input tensor (B, in_channels, T).
            d (List): F0-dependent dilation factors [(B, 1, T) x num_upsamples].

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """

        # currently, same input feature is input to each network
        c = self.input_conv(c)
        e = c

        # source-network forward
        x = self.sn["emb"](x)
        embs = [x]
        for i in range(self.num_upsamples - 1):
            x = self.sn["downsamples"][i](x)
            embs += [x]
        for i in range(self.num_upsamples):
            # excitation generation network
            e = self.sn["upsamples"][i](e) + embs[-i - 1]
            e = self.sn["blocks"][i](e, d[i])
        e_ = self.sn["output_conv"](e)

        # filter-network forward
        embs = [e]
        for i in range(self.num_upsamples - 1):
            if self.share_downsamples:
                e = self.sn["downsamples"][i](e)
            else:
                e = self.fn["downsamples"][i](e)
            embs += [e]
        num_blocks = len(self.filter_network_params["resblock_kernel_sizes"])
        for i in range(self.num_upsamples):
            # resonance filtering network
            if self.share_upsamples:
                c = self.sn["upsamples"][i](c) + embs[-i - 1]
            else:
                c = self.fn["upsamples"][i](c) + embs[-i - 1]
            cs = 0.0  # initialize
            for j in range(num_blocks):
                cs += self.fn["blocks"][i * num_blocks + j](c)
            c = cs / num_blocks
        c = self.fn["output_conv"](c)

        return c, e_

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logger.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f"Weight norm is removed from {m}.")
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)
