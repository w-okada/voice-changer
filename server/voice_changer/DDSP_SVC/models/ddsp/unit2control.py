import gin

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .pcmer import PCmer


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Unit2Control(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, 256)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))

    def forward(self, units, f0, phase, volume, spk_id = None, spk_mix_dict = None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
        x = x + self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch - 1)
            else:
                x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls, x

