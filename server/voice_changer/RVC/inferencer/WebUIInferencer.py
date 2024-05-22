import torch
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.rvc_models.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


class WebUIInferencer(Inferencer):
    def loadModel(self, file: str, gpu: int):
        self.setProps(EnumInferenceTypes.pyTorchWebUI, file, True, gpu)

        dev = DeviceManager.get_instance().getDevice(gpu)
        isHalf = DeviceManager.get_instance().halfPrecisionAvailable(gpu)

        cpt = torch.load(file, map_location="cpu")
        model = SynthesizerTrnMsNSFsidM(**cpt["params"], is_half=isHalf)

        model.eval()
        model.load_state_dict(cpt["weight"], strict=False)

        model = model.to(dev)
        if isHalf:
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
        skip_head: torch.Tensor | None,
        return_length: torch.Tensor | None,
    ) -> torch.Tensor:
        res = self.model.infer(feats, pitch_length, pitch, pitchf, sid, skip_head=skip_head)
        res = res[0][0, 0].to(dtype=torch.float32)
        res = torch.clip(res, -1.0, 1.0)
        return res
