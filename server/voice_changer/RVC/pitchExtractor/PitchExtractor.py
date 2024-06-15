from typing import Protocol
import torch
import numpy as np
from typing import Any


class PitchExtractor(Protocol):
    type: str

    def extract(self, audio: torch.Tensor | np.ndarray[Any, np.float32], pitchf: torch.Tensor | np.ndarray[Any, np.float32], f0_up_key: int, sr: int, window: int) -> tuple[np.ndarray[Any, np.int64], np.ndarray[Any, np.float32]]:
        ...

    def getPitchExtractorInfo(self):
        return {
            "pitchExtractorType": self.type,
        }
