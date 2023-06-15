from dataclasses import dataclass
from typing import Any


@dataclass
class FilePaths:
    configFilename: str | None
    pyTorchModelFilename: str | None
    onnxModelFilename: str | None
    clusterTorchModelFilename: str | None
    featureFilename: str | None
    indexFilename: str | None
