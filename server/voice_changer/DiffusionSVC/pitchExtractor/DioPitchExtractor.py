import pyworld
import numpy as np
from const import PitchExtractorType
import torch

from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor


class DioPitchExtractor(PitchExtractor):

    def __init__(self):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "dio"
        self.f0_min = 50
        self.f0_max = 1100
        self.sapmle_rate = 16000
        self.uv_interp = True

    def extract(self, audio: torch.Tensor, pitch, f0_up_key, window, silence_front=0):
        audio = audio.detach().cpu().numpy()
        start_frame = int(silence_front * self.sapmle_rate / window)
        real_silence_front = start_frame * window / self.sapmle_rate
        audio = audio[int(np.round(real_silence_front * self.sapmle_rate)):]

        _f0, t = pyworld.dio(
            audio.astype(np.double),
            self.sapmle_rate,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            channels_in_octave=2,
            frame_period=(1000 * window / self.sapmle_rate)
        )
        f0 = pyworld.stonemask(audio.astype(np.double), _f0, t, self.sapmle_rate)
        pitch[-f0.shape[0]:] = f0[:pitch.shape[0]]
        f0 = pitch

        if self.uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        f0 = f0 * 2 ** (float(f0_up_key) / 12)

        return f0
