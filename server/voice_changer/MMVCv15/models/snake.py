# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Snake Activation Function Module.

References:
    - Neural Networks Fail to Learn Periodic Functions and How to Fix It
        https://arxiv.org/pdf/2006.08195.pdf
    - BigVGAN: A Universal Neural Vocoder with Large-Scale Training
        https://arxiv.org/pdf/2206.04658.pdf

"""

import torch
import torch.nn as nn


class Snake(nn.Module):
    """Snake activation function module."""

    def __init__(self, channels, init=50):
        """Initialize Snake module.

        Args:
            channels (int): Number of feature channels.
            init (float): Initial value of the learnable parameter alpha.
                          According to the original paper, 5 ~ 50 would be
                          suitable for periodic data (i.e. voices).

        """
        super(Snake, self).__init__()
        alpha = init * torch.ones(1, channels, 1)
        self.alpha = nn.Parameter(alpha)

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        return x + torch.sin(self.alpha * x) ** 2 / self.alpha
