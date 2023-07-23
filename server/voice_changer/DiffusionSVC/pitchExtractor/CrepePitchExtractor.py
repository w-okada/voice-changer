import torchcrepe
import torch
import numpy as np
from const import PitchExtractorType
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.utils.VoiceChangerModel import AudioInOut


class CrepePitchExtractor(PitchExtractor):

    def __init__(self, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "crepe"
        self.f0_min = 50
        self.f0_max = 1100
        self.uv_interp = True
        self.device = DeviceManager.get_instance().getDevice(gpu)

    def extract(self, audio: AudioInOut, sr: int, block_size: int, model_sr: int, pitch, f0_up_key, silence_front=0):
        hop_size = block_size * sr / model_sr
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        offset_frame_number = silence_front * 16000
        start_frame = int(offset_frame_number / hop_size)  # frame
        real_silence_front = start_frame * hop_size / 16000  # 秒
        audio_t = audio_t[:, int(np.round(real_silence_front * 16000)):]

        f0, pd = torchcrepe.predict(
            audio_t,
            sr,
            hop_length=hop_size,
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
