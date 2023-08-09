from dataclasses import dataclass

from const import VoiceChangerType
from typing import Literal, TypeAlias

LoadModelParamFileKind: TypeAlias = Literal[
    "mmvcv13Config",
    "mmvcv13Model",
    "mmvcv15Config",
    "mmvcv15Model",
    "mmvcv15Correspondence",
    "soVitsSvc40Config",
    "soVitsSvc40Model",
    "soVitsSvc40Cluster",
    "rvcModel",
    "rvcIndex",
    "ddspSvcModel",
    "ddspSvcModelConfig",
    "ddspSvcDiffusion",
    "ddspSvcDiffusionConfig",
    "diffusionSVCModel",
    "beatriceModel",
]


@dataclass
class LoadModelParamFile:
    name: str
    kind: LoadModelParamFileKind
    dir: str


@dataclass
class LoadModelParams:
    voiceChangerType: VoiceChangerType
    slot: int
    isSampleMode: bool
    sampleId: str
    files: list[LoadModelParamFile]
    params: dict
