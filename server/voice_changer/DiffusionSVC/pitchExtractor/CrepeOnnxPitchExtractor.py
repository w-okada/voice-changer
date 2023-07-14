import numpy as np
from const import PitchExtractorType
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
import onnxruntime
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

    def extract(self, audio, pitchf, f0_up_key, sr, window, silence_front=0):
        n_frames = int(len(audio) // window) + 1
        start_frame = int(silence_front * sr / window)
        real_silence_front = start_frame * window / sr

        silence_front_offset = int(np.round(real_silence_front * sr))
        audio = audio[silence_front_offset:]

        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        precision = 10.0

        audio_num = audio.cpu()
        onnx_f0, onnx_pd = onnxcrepe.predict(
            self.onnx_session,
            audio_num,
            sr,
            precision=precision,
            fmin=f0_min,
            fmax=f0_max,
            batch_size=256,
            return_periodicity=True,
            decoder=onnxcrepe.decode.weighted_argmax,
            )

        f0 = onnxcrepe.filter.median(onnx_f0, 3)
        pd = onnxcrepe.filter.median(onnx_pd, 3)

        f0[pd < 0.1] = 0
        f0 = f0.squeeze()

        f0 *= pow(2, f0_up_key / 12)
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0bak = pitchf.copy()
        f0_mel = 1127.0 * np.log(1.0 + f0bak / 700.0)
        f0_mel = np.clip(
            (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0, 1.0, 255.0
        )
        pitch_coarse = f0_mel.astype(int)

        return pitch_coarse, pitchf
