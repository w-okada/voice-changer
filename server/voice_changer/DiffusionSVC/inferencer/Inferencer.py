from typing import Any, Protocol
import torch
import onnxruntime

from const import DiffusionSVCInferenceType


class Inferencer(Protocol):
    inferencerType: DiffusionSVCInferenceType = "combo"
    file: str
    isHalf: bool = True
    gpu: int = 0

    model: onnxruntime.InferenceSession | Any | None = None

    def loadModel(self, file: str, gpu: int):
        ...

    def getConfig(self) -> tuple[int, int]:
        ...

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def setProps(
        self,
        inferencerType: DiffusionSVCInferenceType,
        file: str,
        isHalf: bool,
        gpu: int,
    ):
        self.inferencerType = inferencerType
        self.file = file
        self.isHalf = isHalf
        self.gpu = gpu

    def getInferencerInfo(self):
        return {
            "inferencerType": self.inferencerType,
            "file": self.file,
            "isHalf": self.isHalf,
            "gpu": self.gpu,
        }
