import numpy as np
import torch
import onnxruntime

from const import PitchExtractorType
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.TorchUtils import circular_write

class FcpeOnnxPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.type: PitchExtractorType = "fcpe_onnx"
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().get_onnx_execution_provider()

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        self.onnx_session = onnxruntime.InferenceSession(file, sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)

    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key: int, sr: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented()
