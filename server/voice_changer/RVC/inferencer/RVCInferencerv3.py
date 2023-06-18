import torch
from torch import device

from const import EnumInferenceTypes
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from .model_v3.models import SynthesizerTrnMs256NSFSid


class RVCInferencerv3(Inferencer):
    def loadModel(self, file: str, gpu: device):
        print("nadare v3 load start")
        super().setProps(EnumInferenceTypes.pyTorchRVCv3, file, True, gpu)

        dev = DeviceManager.get_instance().getDevice(gpu)
        isHalf = False # DeviceManager.get_instance().halfPrecisionAvailable(gpu)

        cpt = torch.load(file, map_location="cpu")
        model = SynthesizerTrnMs256NSFSid(**cpt["params"])

        model.eval()
        model.load_state_dict(cpt["weight"], strict=False)

        model = model.to(dev)
        if isHalf:
            model = model.half()

        self.model = model
        print("load model comprete")
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.infer(feats, pitch_length, pitch, pitchf, sid)
