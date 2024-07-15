import torch
from const import EnumInferenceTypes

from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs256NSFsid_nono


class RVCInferencerNono(Inferencer):
    def load_model(self, file: str):
        self.set_props(EnumInferenceTypes.pyTorchRVCNono, file)

        device_manager = DeviceManager.get_instance()
        is_half = device_manager.use_fp16()

        cpt = torch.load(file, map_location="cpu")
        model = SynthesizerTrnMs256NSFsid_nono(*cpt["config"], is_half=is_half)

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
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        skip_head: int,
        return_length: int,
        formant_length: int,
    ) -> torch.Tensor:
        res = self.model.infer(
            feats,
            pitch_length,
            sid,
            skip_head=skip_head,
            return_length=return_length,
            formant_length=formant_length
        )
        res = res[0][0, 0]
        return torch.clip(res, -1.0, 1.0)
