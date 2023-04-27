from dataclasses import dataclass


@dataclass
class FilePaths:
    configFilename: str
    pyTorchModelFilename: str
    onnxModelFilename: str
    clusterTorchModelFilename: str
    featureFilename: str
    indexFilename: str


@dataclass
class LoadModelParams:
    slot: int
    isHalf: bool
    files: FilePaths
    params: str
