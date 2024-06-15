import torch
from torch import device

from const import EnumInferenceTypes
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from .voras_beta.models import Synthesizer


class VoRASInferencer(Inferencer):
    def load_model(self, file: str):
        super().set_props(EnumInferenceTypes.pyTorchVoRASbeta, file)

        dev = DeviceManager.get_instance().device
        self.isHalf = False  # DeviceManager.get_instance().is_fp16_available(gpu)

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
        convert_length: int | None,
    ) -> torch.Tensor:
        return self.model.infer(feats, pitch_length, pitch, pitchf, sid)
