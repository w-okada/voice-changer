from dataclasses import dataclass


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
    files: FilePaths
    params: str
