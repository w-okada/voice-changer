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
        self.sapmle_rate = 16000
        self.uv_interp = True
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            self.device = torch.device("cpu")

        self.rmvpe = RMVPE(model_path=file, is_half=False, device=self.device)

    def extract(self, audio: torch.Tensor, pitch, f0_up_key, window, silence_front=0):
        start_frame = int(silence_front * self.sapmle_rate / window)
        real_silence_front = start_frame * window / self.sapmle_rate
        audio = audio[int(np.round(real_silence_front * self.sapmle_rate)):]

        print("[RMVPE AUDI]", audio.device)
        print("[RMVPE RMVPE]", self.rmvpe.device)

        f0 = self.rmvpe.infer_from_audio_t(audio, thred=0.03)
        # f0, pd = torchcrepe.predict(
        #     audio.unsqueeze(0),
        #     self.sapmle_rate,
        #     hop_length=window,
        #     fmin=self.f0_min,
        #     fmax=self.f0_max,
        #     # model="tiny",
        #     model="full",
        #     batch_size=256,
        #     decoder=torchcrepe.decode.weighted_argmax,
        #     device=self.device,
        #     return_periodicity=True,
        # )
        # f0 = torchcrepe.filter.median(f0, 3)  # 本家だとmeanですが、harvestに合わせmedianフィルタ
        # pd = torchcrepe.filter.median(pd, 3)
        # f0[pd < 0.1] = 0
        # f0 = f0.squeeze()
        pitch[-f0.shape[0]:] = f0[:pitch.shape[0]]
        f0 = pitch

        if self.uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min

        f0 = f0 * 2 ** (float(f0_up_key) / 12)

        return f0
