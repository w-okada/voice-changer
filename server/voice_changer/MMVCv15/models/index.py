# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Indexing-related functions."""

import torch
from torch.nn import ConstantPad1d as pad1d


def pd_indexing(x, d, dilation, batch_index, ch_index):
    """Pitch-dependent indexing of past and future samples.

    Args:
        x (Tensor): Input feature map (B, C, T).
        d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        dilation (Int): Dilation size.
        batch_index (Tensor): Batch index
        ch_index (Tensor): Channel index

    Returns:
        Tensor: Past output tensor (B, out_channels, T)
        Tensor: Future output tensor (B, out_channels, T)

    """
    (_, _, batch_length) = d.size()
    dilations = d * dilation

    # get past index
    idxP = torch.arange(-batch_length, 0).float()
    idxP = idxP.to(x.device)
    idxP = torch.add(-dilations, idxP)
    idxP = idxP.round().long()
    maxP = -((torch.min(idxP) + batch_length))
    assert maxP >= 0
    idxP = (batch_index, ch_index, idxP)
    # padding past tensor
    xP = pad1d((maxP, 0), 0)(x)

    # get future index
    idxF = torch.arange(0, batch_length).float()
    idxF = idxF.to(x.device)
    idxF = torch.add(dilations, idxF)
    idxF = idxF.round().long()
    maxF = torch.max(idxF) - (batch_length - 1)
    assert maxF >= 0
    idxF = (batch_index, ch_index, idxF)
    # padding future tensor
    xF = pad1d((0, maxF), 0)(x)

    return xP[idxP], xF[idxF]


def index_initial(n_batch, n_ch, tensor=True):
    """Tensor batch and channel index initialization.

    Args:
        n_batch (Int): Number of batch.
        n_ch (Int): Number of channel.
        tensor (bool): Return tensor or numpy array

    Returns:
        Tensor: Batch index
        Tensor: Channel index

    """
    batch_index = []
    for i in range(n_batch):
        batch_index.append([[i]] * n_ch)
    ch_index = []
    for i in range(n_ch):
        ch_index += [[i]]
    ch_index = [ch_index] * n_batch

    if tensor:
        batch_index = torch.tensor(batch_index)
        ch_index = torch.tensor(ch_index)
        if torch.cuda.is_available():
            batch_index = batch_index.cuda()
            ch_index = ch_index.cuda()
    return batch_index, ch_index
