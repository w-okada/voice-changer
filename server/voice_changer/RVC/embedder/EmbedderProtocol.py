from typing import Protocol

import torch


class EmbedderProtocol(Protocol):

    def load_model(self, file: str):
        ...

    def extract_features(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        ...
