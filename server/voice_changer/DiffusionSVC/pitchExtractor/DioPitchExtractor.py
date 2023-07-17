import pyworld
import numpy as np
from const import PitchExtractorType

from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.utils.VoiceChangerModel import AudioInOut


class DioPitchExtractor(PitchExtractor):

    def __init__(self):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "dio"
        self.f0_min = 50
        self.f0_max = 1100
        # self.sapmle_rate = 44100
        # self.sapmle_rate = 16000
        self.uv_interp = True

    def extract(self, audio: AudioInOut, sr: int, block_size: int, model_sr: int, pitch, f0_up_key, silence_front=0):
        silence_front: int = 0.  # TODO: chunkサイズが小さいときに音程を取れなくなる対策
        hop_size = block_size * sr / model_sr

        offset_frame_number = silence_front * sr
        start_frame = int(offset_frame_number / hop_size)  # frame
        real_silence_front = start_frame * hop_size / sr  # 秒
        audio = audio[int(np.round(real_silence_front * sr)):]

        _f0, t = pyworld.dio(
            audio.astype(np.double),
            sr,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            channels_in_octave=2,
            frame_period=(1000 * hop_size / sr)
        )
        f0 = pyworld.stonemask(audio.astype(np.double), _f0, t, sr)
        pitch[-f0.shape[0]:] = f0[:pitch.shape[0]]
        f0 = pitch

        if self.uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        f0 = f0 * 2 ** (float(f0_up_key) / 12)

        return f0
