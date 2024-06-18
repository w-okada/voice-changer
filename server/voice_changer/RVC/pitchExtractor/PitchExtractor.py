from typing import Protocol
import torch
import numpy as np
from typing import Any


class PitchExtractor(Protocol):
    type: str

    def extract(
        self,
        audio: torch.Tensor,
        sr: int,
        window: int,
    ) -> torch.Tensor:
        ...

    def getPitchExtractorInfo(self):
        return {
            "pitchExtractorType": self.type,
        }
