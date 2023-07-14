import numpy as np
import torch
import torch.nn as nn

from voice_changer.utils.VoiceChangerModel import AudioInOut


class VolumeExtractor:

    def __init__(self, hop_size: float):
        self.hop_size = hop_size

    def getVolumeExtractorInfo(self):
        return {
            "hop_size": self.hop_size
        }

    def extract(self, audio: torch.Tensor):
        audio = audio.squeeze().cpu()
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
        volume = np.array(
            [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume

    def extract_t(self, audio: torch.Tensor):
        with torch.no_grad():
            audio = audio.squeeze()
            n_frames = int(audio.size(0) // self.hop_size) + 1
            audio2 = audio ** 2

            audio2_frames = audio2.unfold(0, int(self.hop_size), int(self.hop_size)).contiguous()

            volume = torch.mean(audio2_frames, dim=-1)
            volume = torch.sqrt(volume)
            if volume.size(0) < n_frames:
                volume = torch.nn.functional.pad(volume, (0, n_frames - volume.size(0)), 'constant', volume[-1])
            return volume

    def get_mask_from_volume(self, volume, block_size: int, threshold=-60.0, device='cpu') -> torch.Tensor:
        volume = volume.cpu().numpy()
        mask = (volume > 10 ** (float(threshold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, block_size).squeeze(-1)
        return mask

    def get_mask_from_volume_t(self, volume: torch.Tensor, block_size: int, threshold=-60.0, device='cpu') -> torch.Tensor:
        volume = volume.squeeze()
        mask = (volume > 10.0 ** (float(threshold) / 20)).float()
        mask = nn.functional.pad(mask, (4, 0), 'constant', mask[0])
        mask = nn.functional.pad(mask, (0, 4), 'constant', mask[-1])
        mask = torch.max(mask.unfold(-1, 9, 1), -1)[0]
        mask = mask.to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, block_size).squeeze(-1)
        print("[get_mask_from_volume_t 3]", mask.shape)
        return mask


def upsample(signal: torch.Tensor, factor: int) -> torch.Tensor:
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal, signal[:, :, -1:]), 2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)
