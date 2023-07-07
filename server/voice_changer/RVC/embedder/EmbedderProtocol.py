from typing import Protocol

import torch
from torch import device


class EmbedderProtocol(Protocol):

    def loadModel(self, file: str, dev: device, isHalf: bool = True):
        ...

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        ...
