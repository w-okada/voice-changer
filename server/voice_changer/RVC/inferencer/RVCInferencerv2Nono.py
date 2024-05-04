import torch
import json
from safetensors import safe_open
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid_nono
from voice_changer.common.SafetensorsUtils import load_model


class RVCInferencerv2Nono(Inferencer):
    def loadModel(self, file: str, gpu: int):
        dev = DeviceManager.get_instance().getDevice(gpu)
        #isHalf = DeviceManager.get_instance().halfPrecisionAvailable(gpu)
        isHalf = False
        self.setProps(EnumInferenceTypes.pyTorchRVCv2Nono, file, isHalf, gpu)

        # Keep torch.load for backward compatibility, but discourage the use of this loading method
        if '.safetensors' in file:
            with safe_open(file, 'pt', device=str(dev)) as cpt:
                config = json.loads(cpt.metadata()['config'])
                model = SynthesizerTrnMs768NSFsid_nono(*config, is_half=False).to(dev)
                load_model(model, cpt, strict=False)
        else:
            cpt = torch.load(file, map_location=dev)
            model = SynthesizerTrnMs768NSFsid_nono(*cpt["config"], is_half=False).to(dev)
            model.load_state_dict(cpt["weight"], strict=False)

        model.eval().remove_weight_norm()

        self.model = model
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        skip_head: torch.Tensor | None,
        return_length: torch.Tensor | None,
    ) -> torch.Tensor:
        res = self.model.infer(feats, pitch_length, sid, skip_head=skip_head, return_length=return_length)
        res = res[0][0, 0].to(dtype=torch.float32)
        return torch.clip(res, -1.0, 1.0, out=res)
