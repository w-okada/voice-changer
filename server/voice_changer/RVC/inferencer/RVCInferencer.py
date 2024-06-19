import torch
from const import EnumInferenceTypes

from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs256NSFsid


class RVCInferencer(Inferencer):
    def load_model(self, file: str):
        self.set_props(EnumInferenceTypes.pyTorchRVC, file)

        device_manager = DeviceManager.get_instance()
        is_half = device_manager.use_fp16()

        cpt = torch.load(file, map_location="cpu")
        model = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)

        model.eval()
        model.load_state_dict(cpt["weight"], strict=False)

        model = model.to(device_manager.device)
        if is_half:
            model = model.half()

        self.model = model
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        skip_head: int | None,
    ) -> torch.Tensor:
        res = self.model.infer(feats, pitch_length, pitch, pitchf, sid, skip_head=skip_head)
        res = res[0][0, 0].float()
        return torch.clip(res, -1.0, 1.0)
