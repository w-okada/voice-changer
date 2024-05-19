import numpy as np
from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.rmvpe.rmvpe import RMVPE
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.TorchUtils import circular_write
import torch


class RMVPEPitchExtractor(PitchExtractor):

    def __init__(self, file: str, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "rmvpe"
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # self.uv_interp = True
        # self.input_sr = -1
        self.device = DeviceManager.get_instance().getDevice(gpu)
        # isHalf = DeviceManager.get_instance().halfPrecisionAvailable(gpu)
        isHalf = False
        self.rmvpe = RMVPE(model_path=file, is_half=isHalf, device=self.device)

    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key: int, sr: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        f0: torch.Tensor = self.rmvpe.infer_from_audio_t(audio, threshold=0.03)

        f0 *= 2 ** (f0_up_key / 12)
        circular_write(f0, pitchf)
        f0_mel = 1127.0 * torch.log(1.0 + pitchf / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).to(dtype=torch.int64)
        return f0_coarse.unsqueeze(0), pitchf.unsqueeze(0)
