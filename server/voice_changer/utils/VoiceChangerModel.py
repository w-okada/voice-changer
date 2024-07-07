from typing import Any, Protocol, TypeAlias
import torch
import numpy as np
from const import VoiceChangerType

from voice_changer.utils.LoadModelParams import LoadModelParams


AudioInOut: TypeAlias = np.ndarray[Any, np.dtype[np.int16]]
AudioInOutFloat: TypeAlias = np.ndarray[Any, np.dtype[np.float32]]

PitchfInOut: TypeAlias = np.ndarray[Any, np.dtype[np.int16]]
FeatureInOut: TypeAlias = np.ndarray[Any, np.dtype[np.int16]]


class VoiceChangerModel(Protocol):
    voiceChangerType: VoiceChangerType = "RVC"

    # load_model: Callable[..., dict[str, Any]]
    def load_model(self, params: LoadModelParams):
        ...

    def get_processing_sampling_rate(self) -> int:
        ...

    def get_info(self) -> dict[str, Any]:
        ...

    def convert(self, data: torch.Tensor, sample_rate: int) -> torch.Tensor:
        ...

    def inference(self, data: tuple[Any, ...]) -> torch.Tensor:
        ...

    def generate_input(
        self,
        newData: AudioInOut,
        inputSize: int,
        crossfadeSize: int,
        solaSearchFrame: int,
    ) -> tuple[Any, ...]:
        ...

    def update_settings(self, key: str, val: Any, old_val: Any):
        ...

    def setSamplingRate(self, inputSampleRate: int, outputSampleRate: int):
        ...

    def realloc(self, block_frame: int, extra_frame: int, crossfade_frame: int, sola_search_frame: int):
        ...