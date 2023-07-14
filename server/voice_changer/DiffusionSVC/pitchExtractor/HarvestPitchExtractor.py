import pyworld
import numpy as np
from const import PitchExtractorType
import torch
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class HarvestPitchExtractor(PitchExtractor):

    def __init__(self):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "harvest"
        self.f0_min = 50
        self.f0_max = 1100
        self.sapmle_rate = 16000
        self.uv_interp = True

    def extract(self, audio: torch.Tensor, pitchf, f0_up_key, sr, window, silence_front=0):
        audio = audio.detach().cpu().numpy()
        start_frame = int(silence_front * self.sapmle_rate / window)
        real_silence_front = start_frame * window / self.sapmle_rate
        audio = audio[int(np.round(real_silence_front * self.sapmle_rate)):]
        f0, _ = pyworld.harvest(
            audio.astype('double'),
            16000,
            f0_floor=50,
            f0_ceil=1100,
            frame_period=(1000 * window / self.sapmle_rate))
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0 = pitchf

        if self.uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < 50] = 50

        f0 = f0 * 2 ** (float(f0_up_key) / 12)

        return f0
