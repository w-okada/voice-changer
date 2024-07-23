from const import PitchExtractorType
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.rmvpe.rmvpe import RMVPE
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
import torch


class RMVPEPitchExtractor(PitchExtractor):

    def __init__(self, file: str):
        super().__init__()
        self.type: PitchExtractorType = "rmvpe"

        device_manager = DeviceManager.get_instance()
        self.rmvpe = RMVPE(model_path=file, is_half=device_manager.use_fp16(), device=device_manager.device)

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        return self.rmvpe.infer_from_audio_t(audio).squeeze()