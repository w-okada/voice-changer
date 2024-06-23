from typing import Any, Protocol
import torch
import onnxruntime

from const import EnumInferenceTypes


class Inferencer(Protocol):
    inferencerType: EnumInferenceTypes = EnumInferenceTypes.pyTorchRVC
    file: str

    model: onnxruntime.InferenceSession | Any | None = None

    def load_model(self, file: str):
        ...

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
        ...

    def set_props(
        self,
        inferencerType: EnumInferenceTypes,
        file: str,
    ):
        self.inferencerType = inferencerType
        self.file = file

    def getInferencerInfo(self):
        return {
            "inferencerType": self.inferencerType.value,
            "file": self.file,
        }
