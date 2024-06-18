import numpy as np
import torch
from const import PitchExtractorType, F0_MIN, F0_MAX
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
import onnxruntime
from voice_changer.RVC.pitchExtractor import onnxcrepe


class CrepeOnnxPitchExtractor(PitchExtractor):

    def __init__(self, type: PitchExtractorType, file: str):
        self.type = type
        super().__init__()
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().get_onnx_execution_provider()

        self.onnx_session = onnxruntime.InferenceSession(
            file, providers=onnxProviders, provider_options=onnxProviderOptions
        )

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        # NOTE: Crepe ONNX model is FP32. Conversion was not tested so keeping input in FP32.
        audio_num = audio.float().detach().cpu().numpy()
        onnx_f0, onnx_pd = onnxcrepe.predict(
            self.onnx_session,
            audio_num,
            sr,
            precision=10.0,
            fmin=F0_MIN,
            fmax=F0_MAX,
            batch_size=256,
            return_periodicity=True,
            decoder=onnxcrepe.decode.weighted_argmax,
        )

        f0: np.ndarray = onnxcrepe.filter.median(onnx_f0, 3)
        pd: np.ndarray = onnxcrepe.filter.median(onnx_pd, 3)

        f0[pd < 0.1] = 0
        return torch.as_tensor(f0, dtype=torch.float32, device=audio.device).squeeze()
