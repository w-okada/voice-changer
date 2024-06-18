import numpy as np
import onnxruntime
import torch
from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.MelExtractorFcpe import Wav2MelModule

class FcpeOnnxPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.file = file
        self.type: PitchExtractorType = "fcpe_onnx"

        device_manager = DeviceManager.get_instance()
        # NOTE: FCPE doesn't seem to be behave correctly in FP16 mode.
        # self.is_half = device_manager.use_fp16()
        self.is_half = False
        (
            onnxProviders,
            onnxProviderOptions,
        ) = device_manager.get_onnx_execution_provider()

        model = load_onnx_model(file, self.is_half)

        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32

        self.threshold = np.array(0.006, dtype=self.fp_dtype_np)

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        self.mel_extractor = Wav2MelModule(
            sr=16000,
            n_mels=128,
            n_fft=1024,
            win_size=1024,
            hop_length=160,
            fmin=0,
            fmax=8000,
            clip_val=1e-05,
            is_half=self.is_half
        ).to(device_manager.device)
        self.onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        mel = self.mel_extractor(audio.unsqueeze(0).float())

        if audio.device.type == 'cuda':
            binding = self.onnx_session.io_binding()

            binding.bind_input('mel', device_type='cuda', device_id=audio.device.index, element_type=self.fp_dtype_np, shape=tuple(mel.shape), buffer_ptr=mel.contiguous().data_ptr())
            binding.bind_cpu_input('threshold', self.threshold)

            binding.bind_output('pitchf', device_type='cuda', device_id=audio.device.index)

            self.onnx_session.run_with_iobinding(binding)

            output = [output.numpy() for output in binding.get_outputs()]
        else:
            output: list[np.ndarray] = self.onnx_session.run(
                ["pitchf"],
                {
                    "mel": mel.detach().cpu().numpy(),
                    "threshold": self.threshold,
                },
            )
        # self.onnx_session.end_profiling()

        return torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=audio.device).squeeze()