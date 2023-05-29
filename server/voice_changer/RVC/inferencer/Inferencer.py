from typing import Any, Protocol
import torch
import onnxruntime


class Inferencer(Protocol):
    # inferencerType: EnumInferenceTypes = EnumInferenceTypes.pyTorchRVC
    # file: str
    # isHalf: bool = True
    # dev: device | None
    # onnxProviders: list[str] | None
    # onnxProviderOptions: Any | None
    model: onnxruntime.InferenceSession | Any | None = None

    def loadModel(self, file: str, gpu: int):
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

    # def setProps(
    #     self,
    #     inferencerType: EnumInferenceTypes,
    #     file: str,
    #     dev: device | None,
    #     onnxProviders: list[str] | None,
    #     onnxProviderOptions: Any | None,
    #     isHalf: bool = True,
    # ):
    #     self.inferencerType = inferencerType
    #     self.file = file
    #     self.isHalf = isHalf
    #     self.dev = dev
    #     self.onnxProviders = onnxProviders
    #     self.onnxProviderOptions = onnxProviderOptions
