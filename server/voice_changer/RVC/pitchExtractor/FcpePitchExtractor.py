import torch

from const import PitchExtractorType
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.FCPE import spawn_infer_model_from_pt
from voice_changer.common.MelExtractorFcpe import Wav2MelModule

class FcpePitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.type: PitchExtractorType = "fcpe"
        device_manager = DeviceManager.get_instance()
        # self.is_half = device_manager.use_fp16()
        # NOTE: FCPE doesn't seem to be behave correctly in FP16 mode.
        self.is_half = False

        model = spawn_infer_model_from_pt(file, self.is_half, device_manager.device, bundled_model=True)
        self.mel_extractor = Wav2MelModule(
            sr=16000,
            n_mels=128,
            n_fft=1024,
            win_size=1024,
            hop_length=160,
            fmin=0,
            fmax=8000,
            clip_val=1e-05,
            is_half=self.is_half
        ).to(device_manager.device)
        self.fcpe = model

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        mel: torch.Tensor = self.mel_extractor(audio.unsqueeze(0).float())

        return self.fcpe(
            mel,
            decoder_mode="local_argmax",
            threshold=0.006,
        ).squeeze()
