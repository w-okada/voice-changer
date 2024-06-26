import torch
import torchcrepe
from typing import Literal


def load_model(device: torch.device, file: str, capacity: Literal['full', 'tiny'] = 'full'):
    """Preloads model from disk"""
    # Bind model and capacity
    torchcrepe.infer.capacity = capacity
    torchcrepe.infer.model = torchcrepe.Crepe(capacity)

    # Load weights
    torchcrepe.infer.model.load_state_dict(
        torch.load(file, map_location=device if device.type == 'cuda' else 'cpu'))

    # Place on device
    torchcrepe.infer.model = torchcrepe.infer.model.to(device)

    # Eval mode
    torchcrepe.infer.model.eval()
