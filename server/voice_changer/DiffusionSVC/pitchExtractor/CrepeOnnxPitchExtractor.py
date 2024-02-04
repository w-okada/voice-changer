import numpy as np
from const import PitchExtractorType
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
import onnxruntime

from voice_changer.RVC.pitchExtractor import onnxcrepe
from voice_changer.utils.VoiceChangerModel import AudioInOut


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
        self.sapmle_rate = 16000
        self.uv_interp = True

    def extract(self, audio: AudioInOut, sr: int, block_size: int, model_sr: int, pitch, f0_up_key, silence_front=0):
        hop_size = block_size * sr / model_sr

        offset_frame_number = silence_front * sr
        start_frame = int(offset_frame_number / hop_size)  # frame
        real_silence_front = start_frame * hop_size / sr  # 秒
        audio = audio[int(np.round(real_silence_front * sr)):].astype(np.float32)

        precision = (1000 * hop_size / sr)

        onnx_f0, onnx_pd = onnxcrepe.predict(
            self.onnx_session,
            audio,
            sr,
            precision=precision,
            fmin=self.f0_min,
            fmax=self.f0_max,
            batch_size=256,
            return_periodicity=True,
            decoder=onnxcrepe.decode.weighted_argmax,
            )

        f0 = onnxcrepe.filter.median(onnx_f0, 3)
        pd = onnxcrepe.filter.median(onnx_pd, 3)

        f0[pd < 0.1] = 0
        f0 = f0.squeeze()
        pitch[-f0.shape[0]:] = f0[:pitch.shape[0]]
        f0 = pitch

        if self.uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        f0 = f0 * 2 ** (float(f0_up_key) / 12)

        return f0
