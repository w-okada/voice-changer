from typing import Any, Protocol

import torch
from torch import device

from const import EnumEmbedderTypes


class Embedder(Protocol):
    embedderType: EnumEmbedderTypes = EnumEmbedderTypes.hubert
    file: str
    isHalf: bool = True
    dev: device

    model: Any | None = None

    def loadModel(self, file: str, dev: device, isHalf: bool = True):
        ...

    def extractFeatures(self, feats: torch.Tensor, embChannels=256) -> torch.Tensor:
        ...

    def setProps(
        self,
        embedderType: EnumEmbedderTypes,
        file: str,
        dev: device,
        isHalf: bool = True,
    ):
        self.embedderType = embedderType
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

    def matchCondition(self, embedderType: EnumEmbedderTypes, file: str) -> bool:
        # Check Type
        if self.embedderType != embedderType:
            print(
                "[Voice Changer] embeder type is not match",
                self.embedderType,
                embedderType,
            )
            return False

        # Check File Path
        if self.file != file:
            print(
                "[Voice Changer] embeder file is not match",
                self.file,
                file,
            )
            return False

        else:
            return True

    def to(self, dev: torch.device):
        if self.model is not None:
            self.model = self.model.to(dev)
        return self

    def printDevice(self):
        print("embedder device:", self.model.device)
