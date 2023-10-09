# 実験用。使用しない。

import torch
from torchcrepe.model import Crepe
import os
CENTS_PER_BIN = 20
PITCH_BINS = 360


class TorchCrepe2(torch.nn.Module):

    def __init__(self, model='full'):
        super().__init__()
        self.crepe = Crepe(model)
        file = os.path.join(os.path.dirname(__file__), f'{model}.pth')
        self.crepe.load_state_dict(torch.load(file, map_location="cpu"))
        self.crepe = self.crepe.to(torch.device("cpu"))
        self.crepe.eval()

        self.sample_rate = 16000
        self.hop_length = 160
        self.window_size = 1024

    def forward(self, audio, f0_min: int, f0_max: int):
        # total_frames = 1 + int(audio.size(1) // self.hop_length)
        audio = torch.nn.functional.pad(audio, (self.window_size // 2, self.window_size // 2))
        # batch_size = total_frames

        start = 0
        end = audio.size(1)

        # Chunk
        frames = torch.nn.functional.unfold(
            audio[:, None, None, start:end],
            kernel_size=(1, self.window_size),
            stride=(1, self.hop_length))

        frames = frames.transpose(1, 2).reshape(-1, self.window_size)

        # Place on device
        # frames = frames.to(device)

        # Mean-center
        frames -= frames.mean(dim=1, keepdim=True)

        # Scale
        frames /= torch.max(torch.tensor(1e-10, device=frames.device),
                            frames.std(dim=1, keepdim=True))

        probabilities = self.crepe(frames.to(torch.device("cpu")))
        probabilities = probabilities.reshape(
            audio.size(0), -1, PITCH_BINS).transpose(1, 2)
        
        minidx = frequency_to_bins(torch.tensor(f0_min))
        maxidx = frequency_to_bins(torch.tensor(f0_max), torch.ceil)

        probabilities[:, :minidx] = -float('inf')
        probabilities[:, maxidx:] = -float('inf')

        bins, pitch = weighted_argmax(probabilities)

        return pitch, periodicity(probabilities, bins)


def weighted_argmax(logits):
    """Sample observations using weighted sum near the argmax"""
    # Find center of analysis window
    bins = logits.argmax(dim=1)

    # Find bounds of analysis window
    start = torch.max(torch.tensor(0, device=logits.device), bins - 4)
    end = torch.min(torch.tensor(logits.size(1), device=logits.device), bins + 5)

    # Mask out everything outside of window
    for batch in range(logits.size(0)):
        for time in range(logits.size(2)):
            logits[batch, :start[batch, time], time] = -float('inf')
            logits[batch, end[batch, time]:, time] = -float('inf')

    # Construct weights
    if not hasattr(weighted_argmax, 'weights'):
        weights = bins_to_cents(torch.arange(360))
        weighted_argmax.weights = weights[None, :, None]

    # Ensure devices are the same (no-op if they are)
    weighted_argmax.weights = weighted_argmax.weights.to(logits.device)

    # Convert to probabilities
    with torch.no_grad():
        probs = torch.sigmoid(logits)

    # Apply weights
    cents = (weighted_argmax.weights * probs).sum(dim=1) / probs.sum(dim=1)

    # Convert to frequency in Hz
    return bins, cents_to_frequency(cents)


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    return dither(cents)


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    # noise = scipy.stats.triang.rvs(c=0.5,
    #                                loc=-CENTS_PER_BIN,
    #                                scale=2 * CENTS_PER_BIN,
    #                                size=cents.size())
        
    # 三角分布のtorch書き換え。c=0.5の時のみ正確な値。それ以外は近似値
    c = 0.5
    loc = -CENTS_PER_BIN
    scale = 2 * CENTS_PER_BIN
    u = torch.rand(cents.size())
    # f = (c - u) / (scale / 2) if u < c else (u - c) / (scale / 2)
    f = torch.where(u < c, (c - u) / (scale / 2), (u - c) / (scale / 2))
    noise = 2 * scale * ((1 - f.abs()) ** 0.5) + loc
    mask = u >= c
    noise[mask] = 2 * (scale - noise[mask])
    return cents + cents.new_tensor(noise)


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def periodicity(probabilities, bins):
    """Computes the periodicity from the network output and pitch bins"""
    # shape=(batch * time / hop_length, 360)
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)

    # shape=(batch * time / hop_length, 1)
    bins_stacked = bins.reshape(-1, 1).to(torch.int64)

    # Use maximum logit over pitch bins as periodicity
    periodicity = probs_stacked.gather(1, bins_stacked)

    # shape=(batch, time / hop_length)
    return periodicity.reshape(probabilities.size(0), probabilities.size(2))


def cents_to_bins(cents, quantize_fn=torch.floor):
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / CENTS_PER_BIN
    return quantize_fn(bins).int()


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * torch.log2(frequency / 10.)
