from typing import Any, Protocol

import torch
from torch import device

from const import EnumInferenceTypes


class Inferencer(Protocol):
    inferencerType: EnumInferenceTypes = EnumInferenceTypes.pyTorchRVC
    file: str
    isHalf: bool = True
    dev: device

    model: Any | None = None

    def loadModel(self, file: str, dev: device, isHalf: bool = True):
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
        inferencerType: EnumInferenceTypes,
        file: str,
        dev: device,
        isHalf: bool = True,
    ):
        self.inferencerType = inferencerType
        self.file = file
        self.isHalf = isHalf
        self.dev = dev

    def setHalf(self, isHalf: bool):
        self.isHalf = isHalf
        if self.model is not None and isHalf:
            self.model = self.model.half()

    def setDevice(self, dev: device):
        self.dev = dev
        if self.model is not None:
            self.model = self.model.to(self.dev)

    def to(self, dev: torch.device):
        if self.model is not None:
            self.model = self.model.to(dev)
        return self

    def printDevice(self):
        print("inferencer device:", self.model.device)
