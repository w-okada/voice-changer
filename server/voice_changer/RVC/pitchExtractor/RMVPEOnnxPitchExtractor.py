import numpy as np
from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
import onnxruntime
import torch


class RMVPEOnnxPitchExtractor(PitchExtractor):

    def __init__(self, file: str, gpu: int):
        super().__init__()
        self.file = file
        self.pitchExtractorType: PitchExtractorType = "rmvpe_onnx"
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)

        self.device = DeviceManager.get_instance().getDevice(gpu)
        self.threshold = torch.tensor((0.3,), dtype=torch.float32, device=self.device)

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        self.onnx_session = onnxruntime.InferenceSession(self.file, sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)

    def extract(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key: int, sr: int, window: int, silence_front: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        audio = audio.unsqueeze(0)

        if audio.device.type == 'cuda':
            binding = self.onnx_session.io_binding()

            binding.bind_input('waveform', device_type='cuda', device_id=self.device.index, element_type=np.float32, shape=tuple(audio.shape), buffer_ptr=audio.data_ptr())
            binding.bind_input('threshold', device_type='cuda', device_id=self.device.index, element_type=np.float32, shape=tuple(self.threshold.shape), buffer_ptr=self.threshold.data_ptr())

            binding.bind_output('pitchf', device_type='cuda', device_id=self.device.index)

            self.onnx_session.run_with_iobinding(binding)

            output = [output.numpy() for output in binding.get_outputs()]
        else:
            output: list[np.ndarray] = self.onnx_session.run(
                ["pitchf"],
                {
                    "waveform": audio.detach().cpu().numpy(),
                    "threshold": self.threshold.detach().cpu().numpy(),
                },
            )
        # self.onnx_session.end_profiling()

        f0 = torch.as_tensor(output[0], dtype=torch.float32, device=audio.device).squeeze()

        f0 *= 2 ** (f0_up_key / 12)
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0_mel = 1127.0 * torch.log(1.0 + pitchf / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).to(dtype=torch.int64)

        return f0_coarse.unsqueeze(0), pitchf.unsqueeze(0)
