import torchcrepe
import torch
import numpy as np
from const import PitchExtractorType

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class CrepePitchExtractor(PitchExtractor):

    def __init__(self):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "crepe"
        self.f0_min = 50
        self.f0_max = 1100
        self.sapmle_rate = 16000
        self.uv_interp = True
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")

    def extract(self, audio: torch.Tensor, pitch, f0_up_key, window, silence_front=0):
        start_frame = int(silence_front * self.sapmle_rate / window)
        real_silence_front = start_frame * window / self.sapmle_rate
        audio = audio[int(np.round(real_silence_front * self.sapmle_rate)):]

        f0, pd = torchcrepe.predict(
            audio.unsqueeze(0),
            self.sapmle_rate,
            hop_length=window,
            fmin=self.f0_min,
            fmax=self.f0_max,
            # model="tiny",
            model="full",
            batch_size=256,
            decoder=torchcrepe.decode.weighted_argmax,
            device=self.device,
            return_periodicity=True,
        )
        f0 = torchcrepe.filter.median(f0, 3)  # 本家だとmeanですが、harvestに合わせmedianフィルタ
        pd = torchcrepe.filter.median(pd, 3)
        f0[pd < 0.1] = 0
        f0 = f0.squeeze()
        pitch[-f0.shape[0]:] = f0.cpu()[:pitch.shape[0]]
        f0 = pitch

        if self.uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        f0 = f0 * 2 ** (float(f0_up_key) / 12)

        return f0
