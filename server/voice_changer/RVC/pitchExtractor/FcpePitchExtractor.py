import numpy as np
import torchfcpe
import torch

from const import PitchExtractorType
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.TorchUtils import circular_write

class FcpePitchExtractor(PitchExtractor):

    def __init__(self, file: str, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "fcpe"
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        device = DeviceManager.get_instance().getDevice(gpu)

        model = torchfcpe.spawn_infer_model_from_pt(file, device, bundled_model=True)
        # NOTE: FCPE currently cannot be converted to FP16 because MEL extractor expects FP32 input.
        self.fcpe = model

    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key: int, sr: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        f0: torch.Tensor = self.fcpe.infer(
            audio.unsqueeze(0),
            sr=sr,
            decoder_mode="local_argmax",
            threshold=0.006,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
        )
        f0 = f0.squeeze()

        f0 *= 2 ** (f0_up_key / 12)
        circular_write(f0, pitchf)
        f0_mel = 1127.0 * torch.log(1.0 + pitchf / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).to(dtype=torch.int64)
        return f0_coarse.unsqueeze(0), pitchf.unsqueeze(0)