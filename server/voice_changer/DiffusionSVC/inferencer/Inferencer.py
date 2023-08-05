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
        audio_t: torch.Tensor,
        feats: torch.Tensor,
        pitch: torch.Tensor,
        volume: torch.Tensor,
        mask: torch.Tensor,
        sid: torch.Tensor,
        k_step: int,
        infer_speedup: int,
        silence_front: float,
        skip_diffusion: bool = True,
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
