import numpy as np
import torch
from const import PitchExtractorType
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
import onnxruntime
from typing import Any
from voice_changer.RVC.pitchExtractor import onnxcrepe


class CrepeOnnxPitchExtractor(PitchExtractor):

    def __init__(self, pitchExtractorType: PitchExtractorType, file: str, gpu: int):
        self.pitchExtractorType = pitchExtractorType
        super().__init__()
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)

        self.onnx_session = onnxruntime.InferenceSession(
            file, providers=onnxProviders, provider_options=onnxProviderOptions
        )

        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

    def extract(self, audio: torch.Tensor | np.ndarray[Any, np.float32], pitchf: torch.Tensor | np.ndarray[Any, np.float32], f0_up_key: int, sr: int, window: int, silence_front: int = 0):
        start_frame = int(silence_front * sr / window)
        real_silence_front = start_frame * window / sr

        silence_front_offset = int(np.round(real_silence_front * sr))
        audio = audio[silence_front_offset:]

        audio_num = audio.detach().cpu().numpy()
        onnx_f0, onnx_pd = onnxcrepe.predict(
            self.onnx_session,
            audio_num,
            sr,
            precision=10.0,
            fmin=self.f0_min,
            fmax=self.f0_max,
            batch_size=256,
            return_periodicity=True,
            decoder=onnxcrepe.decode.weighted_argmax,
        )

        f0: np.ndarray = onnxcrepe.filter.median(onnx_f0, 3)
        pd: np.ndarray = onnxcrepe.filter.median(onnx_pd, 3)

        f0[pd < 0.1] = 0
        f0 = torch.as_tensor(f0.squeeze(), dtype=torch.float32, device=audio.device)

        f0 *= (f0_up_key / 12) ** 2
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0_mel = 1127.0 * torch.log(1.0 + pitchf / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).to(dtype=torch.int64)
        return f0_coarse.unsqueeze(0), pitchf.unsqueeze(0)
