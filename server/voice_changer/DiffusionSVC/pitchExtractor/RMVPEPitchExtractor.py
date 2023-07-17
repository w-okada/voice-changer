from torchaudio.transforms import Resample
import torch
import numpy as np
from const import PitchExtractorType
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.DiffusionSVC.pitchExtractor.rmvpe.rmvpe import RMVPE
from scipy.ndimage import zoom

from voice_changer.utils.VoiceChangerModel import AudioInOut


class RMVPEPitchExtractor(PitchExtractor):

    def __init__(self, file: str, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "rmvpe"
        self.f0_min = 50
        self.f0_max = 1100
        self.uv_interp = True
        self.input_sr = -1
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")

        self.rmvpe = RMVPE(model_path=file, is_half=False, device=self.device)

    def extract(self, audio: AudioInOut, sr: int, block_size: int, model_sr: int, pitch, f0_up_key, silence_front=0):
        if sr != self.input_sr:
            self.resamle = Resample(sr, 16000, dtype=torch.int16).to(self.device)
            self.input_sr = sr
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        audio_t = self.resamle(audio_t)
        hop_size = 160  # RMVPE固定

        offset_frame_number = silence_front * 16000
        start_frame = int(offset_frame_number / hop_size)  # frame
        real_silence_front = start_frame * hop_size / 16000  # 秒
        audio_t = audio_t[:, int(np.round(real_silence_front * 16000)):]

        f0 = self.rmvpe.infer_from_audio_t(audio_t.squeeze(), thred=0.03)

        desired_hop_size = block_size * 16000 / model_sr
        desired_f0_length = int(audio_t.shape[1] // desired_hop_size) + 1
        resize_factor = desired_f0_length / len(f0)
        f0 = zoom(f0, resize_factor, order=0)

        pitch[-f0.shape[0]:] = f0[:pitch.shape[0]]
        f0 = pitch

        if self.uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        f0 = f0 * 2 ** (float(f0_up_key) / 12)

        return f0
