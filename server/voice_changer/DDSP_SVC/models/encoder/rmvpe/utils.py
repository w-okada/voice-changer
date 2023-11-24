import sys
import numpy as np
import librosa
import torch
from functools import reduce
from .constants import *
from torch.nn.modules.module import _addindent


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count

    
def to_local_average_cents(salience, center=None, thred=0.03):
    """
    find the weighted average cents near the argmax bin
    """

    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # the bin number-to-cents mapping
        to_local_average_cents.cents_mapping = (
                20 * np.arange(N_CLASS) + CONST)

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :], None, thred) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")
    
def to_viterbi_cents(salience, thred=0.03):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_cents, 'transition'):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition

    # Convert to probability
    prob = salience.T
    prob = prob / prob.sum(axis=0)    

    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)

    return np.array([to_local_average_cents(salience[i, :], path[i], thred) for i in
                     range(len(path))])

def to_local_average_f0(hidden, center=None, thred=0.03):
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]  # [B=1, T=1, N]
    idx_cents = idx * 20 + CONST  # [B=1, N]
    if center is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)  # [B, T, 1]
    start = torch.clip(center - 4, min=0)  # [B, T, 1]
    end = torch.clip(center + 5, max=N_CLASS)  # [B, T, 1]
    idx_mask = (idx >= start) & (idx < end)  # [B, T, N]
    weights = hidden * idx_mask  # [B, T, N]
    product_sum = torch.sum(weights * idx_cents, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    cents = product_sum / (weight_sum + (weight_sum == 0))  # avoid dividing by zero, [B, T]
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred  # [B, T]
    f0 = f0 * ~uv
    return f0.squeeze(0).cpu().numpy()

def to_viterbi_f0(hidden, thred=0.03):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_cents, 'transition'):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_cents.transition = transition
    
    # Convert to probability
    prob = hidden.squeeze(0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0) 
    
    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob, to_viterbi_cents.transition).astype(np.int64)
    center = torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device)
    
    return to_local_average_f0(hidden, center=center, thred=thred)
    
        