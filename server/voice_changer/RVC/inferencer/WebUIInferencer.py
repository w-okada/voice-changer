import torch
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.rvc_models.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


class WebUIInferencer(Inferencer):
    def load_model(self, file: str):
        self.set_props(EnumInferenceTypes.pyTorchWebUI, file)

        device_manager = DeviceManager.get_instance()
        dev = device_manager.device
        is_half = device_manager.use_fp16()

        cpt = torch.load(file, map_location="cpu")
        model = SynthesizerTrnMsNSFsidM(**cpt["params"], is_half=is_half)

        model.eval()
        model.load_state_dict(cpt["weight"], strict=False)

        model = model.to(dev)
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
        skip_head: int,
        return_length: int,
        formant_length: int,
    ) -> torch.Tensor:
        assert pitch is not None or pitchf is not None, "Pitch or Pitchf is not found."

        res = self.model.infer(
            feats,
            pitch_length,
            pitch,
            pitchf,
            sid,
            skip_head=skip_head,
            return_length=return_length,
            formant_length=formant_length
        )
        res = res[0][0, 0]
        return torch.clip(res, -1.0, 1.0)
