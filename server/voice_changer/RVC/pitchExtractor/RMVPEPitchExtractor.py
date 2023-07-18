import torch
import numpy as np
from const import PitchExtractorType
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.DiffusionSVC.pitchExtractor.rmvpe.rmvpe import RMVPE


class RMVPEPitchExtractor(PitchExtractor):

    def __init__(self, file: str, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "rmvpe"
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.uv_interp = True
        self.input_sr = -1
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")

        self.rmvpe = RMVPE(model_path=file, is_half=False, device=self.device)

    def extract(self, audio, pitchf, f0_up_key, sr, window, silence_front=0):
        hop_size = 160  # RMVPE固定

        offset_frame_number = silence_front * 16000
        start_frame = int(offset_frame_number / hop_size)  # frame
        real_silence_front = start_frame * hop_size / 16000  # 秒
        audio = audio[int(np.round(real_silence_front * 16000)):]

        f0 = self.rmvpe.infer_from_audio_t(audio, thred=0.03)

        f0 = f0 * 2 ** (float(f0_up_key) / 12)
        pitchf[-f0.shape[0]:] = f0[:pitchf.shape[0]]
        f0 = pitchf

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)
        return f0_coarse, f0bak
