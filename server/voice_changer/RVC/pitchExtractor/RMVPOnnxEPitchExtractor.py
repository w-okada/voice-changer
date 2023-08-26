import numpy as np
from const import PitchExtractorType
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
import onnxruntime


class RMVPOnnxEPitchExtractor(PitchExtractor):

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
        self.onnxProviders = onnxProviders
        self.onnxProviderOptions = onnxProviderOptions

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(self.file, sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)

    def extract(self, audio, pitchf, f0_up_key, sr, window, silence_front=0):
        try:
            # データ変換
            if isinstance(audio, np.ndarray) is False:
                audio = audio = audio.cpu().numpy()

            if isinstance(pitchf, np.ndarray) is False:
                pitchf = pitchf.cpu().numpy().astype(np.float32)

            if audio.ndim != 1:
                raise RuntimeError(f"Exeption in {self.__class__.__name__} audio.ndim is not 1 (size :{audio.ndim}, {audio.shape})")
            if pitchf.ndim != 1:
                raise RuntimeError(f"Exeption in {self.__class__.__name__} pitchf.ndim is not 1 (size :{pitchf.ndim}, {pitchf.shape})")

            # 処理
            silenceFrontFrame = silence_front * sr
            startWindow = int(silenceFrontFrame / window)  # 小数点以下切り捨て
            slienceFrontFrameOffset = startWindow * window
            targetFrameLength = len(audio) - slienceFrontFrameOffset
            minimumFrames = 0.01 * sr
            targetFrameLength = max(minimumFrames, targetFrameLength)
            audio = audio[-targetFrameLength:]
            audio = np.expand_dims(audio, axis=0)

            output = self.onnx_session.run(
                ["f0", "uv"],
                {
                    "waveform": audio.astype(np.float32),
                    "threshold": np.array([0.3]).astype(np.float32),
                },
            )

            f0 = output[0].squeeze()

            f0 *= pow(2, f0_up_key / 12)
            pitchf[-f0.shape[0]:] = f0[: pitchf.shape[0]]

            f0_mel = 1127.0 * np.log(1.0 + pitchf / 700.0)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            f0_coarse = np.rint(f0_mel).astype(int)

        except Exception as e:
            raise RuntimeError(f"Exeption in {self.__class__.__name__}", e)

        return f0_coarse, pitchf
