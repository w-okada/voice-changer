import numpy as np
from const import PitchExtractorType
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
import torchfcpe

class FcpePitchExtractor(PitchExtractor):

    def __init__(self, gpu: int):
        super().__init__()
        self.pitchExtractorType: PitchExtractorType = "fcpe"
        self.device = DeviceManager.get_instance().getDevice(gpu)
        self.fcpe = torchfcpe.spawn_bundled_infer_model(self.device)

    # I merge the code of Voice-Changer-CrepePitchExtractor and RVC-fcpe-infer, sry I don't know how to optimize the function.
    def extract(self, audio, pitchf, f0_up_key, sr, window, silence_front=0):
        start_frame = int(silence_front * sr / window)
        real_silence_front = start_frame * window / sr

        silence_front_offset = int(np.round(real_silence_front * sr))
        audio = audio[silence_front_offset:]

        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0 = self.fcpe.infer(
            audio.to(self.device).unsqueeze(0).float(),
            sr=16000,
            decoder_mode="local_argmax",
            threshold=0.006,
        )
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
