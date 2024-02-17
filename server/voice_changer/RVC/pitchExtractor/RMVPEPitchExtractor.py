import numpy as np
from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.rmvpe.rmvpe import RMVPE
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
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

    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key: int, sr: int, window: int, silence_front: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        offset_frame_number = silence_front * sr
        start_frame = int(offset_frame_number / window)  # frame
        real_silence_front = start_frame * window / sr  # 秒
        audio = audio[int(np.round(real_silence_front * sr)):]

        f0: torch.Tensor = self.rmvpe.infer_from_audio_t(audio, threshold=0.03)

        f0 *= (f0_up_key / 12) ** 2
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0_mel = 1127.0 * torch.log(1.0 + pitchf / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).to(dtype=torch.int64)
        return f0_coarse.unsqueeze(0), pitchf.unsqueeze(0)
