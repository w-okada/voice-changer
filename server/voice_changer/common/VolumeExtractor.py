import numpy as np
import torch
import torch.nn as nn


class VolumeExtractor:
    def __init__(self, hop_size: float, block_size: int, model_sampling_rate: int, audio_sampling_rate: int):
        self.hop_size = hop_size
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.audio_sampling_rate = audio_sampling_rate
        # self.hop_size = self.block_size * self.audio_sampling_rate / self.model_sampling_rate  # モデルの処理単位が512(Diffusion-SVC), 入力のサンプリングレートのサイズにhopsizeを合わせる。

    def extract(self, audio):  # audio: 1d numpy array
        audio = audio.squeeze().cpu()
        print("----VolExtractor2", audio.shape, self.block_size, self.model_sampling_rate, self.audio_sampling_rate, self.hop_size)
        n_frames = int(len(audio) // self.hop_size) + 1
        print("=======> n_frames", n_frames)
        audio2 = audio ** 2
        print("----VolExtractor3", audio2.shape)
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
        print("----VolExtractor4", audio2.shape)
        volume = np.array(
            [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume

    def get_mask_from_volume(self, volume, threhold=-60.0, device='cpu') -> torch.Tensor:
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.block_size).squeeze(-1)
        return mask


def upsample(signal: torch.Tensor, factor: int) -> torch.Tensor:
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal, signal[:, :, -1:]), 2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)
