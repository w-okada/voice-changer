from dataclasses import dataclass
from typing import Any

from const import VoiceChangerType
from typing import Literal, TypeAlias


@dataclass
class FilePaths:
    configFilename: str | None
    pyTorchModelFilename: str | None
    onnxModelFilename: str | None
    clusterTorchModelFilename: str | None
    featureFilename: str | None
    indexFilename: str | None


@dataclass
class LoadModelParams:
    slot: int
    isHalf: bool
    params: Any


LoadModelParamFileKind: TypeAlias = Literal[
    "mmvcv13Config",
    "mmvcv13Model",
    "mmvcv15Config",
    "mmvcv15Model",
    "soVitsSvc40Config",
    "soVitsSvc40Model",
    "soVitsSvc40Cluster",
    "rvcModel",
    "rvcIndex",
    "ddspSvcModel",
    "ddspSvcModelConfig",
    "ddspSvcDiffusion",
    "ddspSvcDiffusionConfig",
]


@dataclass
class LoadModelParamFile:
    name: str
    kind: LoadModelParamFileKind
    dir: str


@dataclass
class LoadModelParams2:
    voiceChangerType: VoiceChangerType
    slot: int
    isSampleMode: bool
    sampleId: str
    files: list[LoadModelParamFile]
