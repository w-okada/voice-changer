import pyworld
import numpy as np
from const import PitchExtractorType

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class DioPitchExtractor(PitchExtractor):

    def __init__(self):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "dio"

    def extract(self, audio, pitchf, f0_up_key, sr, window):
        audio = audio.detach().cpu().numpy()

        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        _f0, t = pyworld.dio(
            audio.astype(np.double),
            sr,
            f0_floor=f0_min,
            f0_ceil=f0_max,
            channels_in_octave=2,
            frame_period=10,
        )
        f0 = pyworld.stonemask(audio.astype(np.double), _f0, t, sr)

        f0 *= 2 ** (f0_up_key / 12)
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0bak = pitchf.copy()
        f0_mel = 1127 * np.log(1 + f0bak / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        pitch_coarse = np.rint(f0_mel).astype(int)

        return pitch_coarse, pitchf
