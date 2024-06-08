from dataclasses import dataclass

from const import VoiceChangerType
from typing import Literal, TypeAlias

LoadModelParamFileKind: TypeAlias = Literal[
    "rvcModel",
    "rvcIndex",
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
