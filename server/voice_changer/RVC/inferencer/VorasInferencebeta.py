import torch
from torch import device

from const import EnumInferenceTypes
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from .voras_beta.models import Synthesizer


class VoRASInferencer(Inferencer):
    def loadModel(self, file: str, gpu: device):
        super().setProps(EnumInferenceTypes.pyTorchVoRASbeta, file, False, gpu)

        dev = DeviceManager.get_instance().getDevice(gpu)
        self.isHalf = False # DeviceManager.get_instance().halfPrecisionAvailable(gpu)

        cpt = torch.load(file, map_location="cpu")
        model = Synthesizer(**cpt["params"])

        model.eval()
        model.load_state_dict(cpt["weight"], strict=False)
        model.remove_weight_norm()
        model.change_speaker(0)

        model = model.to(dev)

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
