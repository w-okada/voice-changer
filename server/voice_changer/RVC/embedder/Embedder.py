from typing import Any

import torch
from torch import device

from const import EmbedderType
from voice_changer.RVC.embedder.EmbedderProtocol import EmbedderProtocol
import logging
logger = logging.getLogger(__name__)

class Embedder(EmbedderProtocol):
    def __init__(self):
        self.embedderType: EmbedderType = "hubert_base"
        self.file: str
        self.dev: device

        self.model: Any | None = None

    def load_model(self, file: str):
        ...

    def extract_features(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        ...

    def get_embedder_info(self):
        return {
            "embedderType": self.embedderType,
            "file": self.file,
        }

    def set_props(
        self,
        embedderType: EmbedderType,
        file: str,
    ):
        self.embedderType = embedderType
        self.file = file

    def matchCondition(self, embedderType: EmbedderType) -> bool:
        return self.embedderType == embedderType
