import torchcrepe
import torch
import numpy as np

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class CrepePitchExtractor(PitchExtractor):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")

    def extract(self, audio, f0_up_key, sr, window, silence_front=0):
        n_frames = int(len(audio) // window) + 1
        start_frame = int(silence_front * sr / window)
        real_silence_front = start_frame * window / sr

        silence_front_offset = int(np.round(real_silence_front * sr))
        audio = audio[silence_front_offset:]

        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0 = torchcrepe.predict(
            torch.tensor(audio).unsqueeze(0),
            sr,
            hop_length=window,
            fmin=f0_min,
            fmax=f0_max,
            # model="tiny",
            model="full",
            batch_size=256,
            decoder=torchcrepe.decode.weighted_argmax,
            device=self.device,
        )
        f0 = f0.squeeze().detach().cpu().numpy()

        f0 = np.pad(
            f0.astype("float"), (start_frame, n_frames - f0.shape[0] - start_frame)
        )

        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        return f0_coarse, f0bak
