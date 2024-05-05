import torchcrepe
import numpy as np
import torch
from typing import Any
from const import PitchExtractorType
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class CrepePitchExtractor(PitchExtractor):

    def __init__(self, gpu: int, model_size: str):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = f"crepe_{model_size}"
        self.model_size = model_size
        self.device = DeviceManager.get_instance().getDevice(gpu)
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)


    def extract(self, audio: torch.Tensor | np.ndarray[Any, np.float32], pitchf: torch.Tensor | np.ndarray[Any, np.float32], f0_up_key: int, sr: int, window: int):
        f0, pd = torchcrepe.predict(
            audio.unsqueeze(0),
            sr,
            hop_length=window,
            fmin=self.f0_min,
            fmax=self.f0_max,
            model=self.model_size,
            batch_size=256,
            decoder=torchcrepe.decode.weighted_argmax,
            device=self.device,
            return_periodicity=True,
        )

        f0: torch.Tensor = torchcrepe.filter.median(f0, 3)  # 本家だとmeanですが、harvestに合わせmedianフィルタ
        pd: torch.Tensor = torchcrepe.filter.median(pd, 3)

        f0[pd < 0.1] = 0
        f0 = f0.squeeze()

        f0 *= 2 ** (f0_up_key / 12)
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0_mel = 1127.0 * torch.log(1.0 + pitchf / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).to(dtype=torch.int64)
        return f0_coarse.unsqueeze(0), pitchf.unsqueeze(0)
