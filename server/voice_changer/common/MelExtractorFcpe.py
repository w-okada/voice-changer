import torch
from .STFT import STFT
from librosa.filters import mel

import logging
logger = logging.getLogger(__name__)

# This module is used by FCPE
# Modules are copied from torchfcpe and modified
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

class Wav2MelModule(torch.nn.Module):
    """
    Wav to mel converter

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): Minimum frequency. Defaults to 0.
        fmax (float, optional): Maximum frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
        mel_type (str, optional): MEL type. Defaults to 'default'.
    """

    def __init__(self,
        sr: int,
        n_mels: int,
        n_fft: int,
        win_size: int,
        hop_length: int,
        fmin: float = None,
        fmax: float = None,
        clip_val: float = 1e-5,
        mel_type="default",
        is_half: bool = False,
    ):
        super().__init__()
        # catch None
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        self.hop_size = hop_length
        if mel_type == "default":
            self.mel_extractor = MelModule(sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val,
                                           out_stft=False, is_half=is_half)
        elif mel_type == "stft":
            self.mel_extractor = MelModule(is_half, sr, n_mels, n_fft, win_size, hop_length, fmin, fmax, clip_val,
                                           out_stft=True, is_half=is_half)
        self.mel_type = mel_type


    @torch.no_grad()
    def __call__(self,
        audio: torch.Tensor,  # (B, T, 1)
    ) -> torch.Tensor:  # (B, T, n_mels)
        """
        Get mel spectrogram

        Args:
            audio (torch.Tensor): Input waveform, shape=(B, T, 1).
            sample_rate (int): Sampling rate.
            keyshift (int, optional): Key shift. Defaults to 0.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        # extract
        mel = self.mel_extractor(audio)
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        if n_frames > int(mel.shape[1]):
            mel = torch.cat((mel, mel[:, -1:, :]), 1)
        if n_frames < int(mel.shape[1]):
            mel = mel[:, :n_frames, :]

        return mel  # (B, T, n_mels)


class MelModule(torch.nn.Module):
    """Mel extractor

    Args:
        sr (int): Sampling rate. Defaults to 16000.
        n_mels (int): Number of mel bins. Defaults to 128.
        n_fft (int): FFT size. Defaults to 1024.
        win_size (int): Window size. Defaults to 1024.
        hop_length (int): Hop length. Defaults to 160.
        fmin (float, optional): Minimum frequency. Defaults to 0.
        fmax (float, optional): Maximum frequency. Defaults to sr/2.
        clip_val (float, optional): Clipping value. Defaults to 1e-5.
    """

    def __init__(
        self,
        sr: int,
        n_mels: int,
        n_fft: int,
        win_size: int,
        hop_length: int,
        fmin: float = None,
        fmax: float = None,
        clip_val: float = 1e-5,
        out_stft: bool = False,
        is_half: bool = False,
    ):
        super().__init__()
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = sr / 2
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.is_half = is_half
        # self.mel_basis = {}
        self.register_buffer(
            'mel_basis',
            torch.tensor(mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)).float(),
            persistent=False
        )
        self.hann_window = torch.nn.ModuleDict()
        self.out_stft = out_stft
        self.stft = STFT(
            filter_length=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window="hann",
        )

    @torch.no_grad()
    def __call__(self,
        y: torch.Tensor,  # (B, T, 1)
    ) -> torch.Tensor:  # (B, T, n_mels)
        """Get mel spectrogram

        Args:
            y (torch.Tensor): Input waveform, shape=(B, T, 1).
            key_shift (int, optional): Key shift. Defaults to 0.
            speed (int, optional): Variable speed enhancement factor. Defaults to 1.
            center (bool, optional): center for torch.stft. Defaults to False.
            no_cache_window (bool, optional): If True will clear cache. Defaults to False.
        return:
            spec (torch.Tensor): Mel spectrogram, shape=(B, T, n_mels).
        """

        y = y.squeeze(-1)

        if torch.min(y) < -1.:
            logger.error(f'min value is {torch.min(y)}')
        if torch.max(y) > 1.:
            logger.error(f'max value is {torch.max(y)}')

        pad_left = (self.win_size - self.hop_length) // 2
        pad_right = max((self.win_size - self.hop_length + 1) // 2, self.win_size - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)

        spec = self.stft.transform(y)

        spec = torch.matmul(self.mel_basis, spec)
        if self.is_half:
            spec = spec.half()
        spec = dynamic_range_compression_torch(spec, clip_val=self.clip_val)
        spec = spec.transpose(-1, -2)
        return spec  # (B, T, n_mels)
