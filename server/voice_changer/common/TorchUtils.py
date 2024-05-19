import torch

def circular_write(new_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    offset = new_data.shape[0]
    target[: -offset] = target[offset :].detach().clone()
    target[-offset :] = new_data
    return target
