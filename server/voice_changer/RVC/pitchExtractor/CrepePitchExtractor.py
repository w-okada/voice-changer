import torchcrepe
import numpy as np
from const import PitchExtractorType
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class CrepePitchExtractor(PitchExtractor):

    def __init__(self, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "crepe"
        self.device = DeviceManager.get_instance().getDevice(gpu)

    def extract(self, audio, pitchf, f0_up_key, sr, window, silence_front=0):
        start_frame = int(silence_front * sr / window)
        real_silence_front = start_frame * window / sr

        silence_front_offset = int(np.round(real_silence_front * sr))
        audio = audio[silence_front_offset:]

        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0, pd = torchcrepe.predict(
            audio.unsqueeze(0),
            sr,
            hop_length=window,
            fmin=f0_min,
            fmax=f0_max,
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

        f0 *= pow(2, f0_up_key / 12)
        pitchf[-f0.shape[0]:] = f0.detach().cpu().numpy()[:pitchf.shape[0]]
        f0bak = pitchf.copy()
        f0_mel = 1127.0 * np.log(1.0 + f0bak / 700.0)
        f0_mel = np.clip(
            (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0, 1.0, 255.0
        )
        pitch_coarse = f0_mel.astype(int)

        return pitch_coarse, pitchf
