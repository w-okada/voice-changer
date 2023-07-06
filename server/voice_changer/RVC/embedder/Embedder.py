from typing import Any, Protocol

import torch
from torch import device

from const import EmbedderType


class Embedder(Protocol):
    embedderType: EmbedderType = "hubert_base"
    file: str
    isHalf: bool = True
    dev: device

    model: Any | None = None

    def loadModel(self, file: str, dev: device, isHalf: bool = True):
        ...

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        ...

    def getEmbedderInfo(self):
        return {
            "embedderType": self.embedderType,
            "file": self.file,
            "isHalf": self.isHalf,
            "devType": self.dev.type,
            "devIndex": self.dev.index,
        }

    def setProps(
        self,
        embedderType: EmbedderType,
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
        elif self.model is not None and isHalf is False:
            self.model = self.model.float()

    def setDevice(self, dev: device):
        self.dev = dev
        if self.model is not None:
            self.model = self.model.to(self.dev)
        return self

    def matchCondition(self, embedderType: EmbedderType) -> bool:
        # Check Type
        if self.embedderType != embedderType:
            print(
                "[Voice Changer] embeder type is not match",
                self.embedderType,
                embedderType,
            )
            return False

        else:
            return True
