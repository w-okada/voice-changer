import pyworld
import numpy as np

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class DioPitchExtractor(PitchExtractor):
    def extract(self, audio, f0_up_key, sr, window, silence_front=0):
        audio = audio.detach().cpu().numpy()
        n_frames = int(len(audio) // window) + 1
        start_frame = int(silence_front * sr / window)
        real_silence_front = start_frame * window / sr

        silence_front_offset = int(np.round(real_silence_front * sr))
        audio = audio[silence_front_offset:]

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
        f0 = np.pad(f0.astype("float"), (start_frame, n_frames - len(f0) - start_frame))

        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak
