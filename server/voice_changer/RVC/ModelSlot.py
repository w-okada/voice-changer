from const import EnumInferenceTypes, EnumEmbedderTypes

from dataclasses import dataclass


@dataclass
class ModelSlot:
    modelFile: str = ""
    indexFile: str = ""
    defaultTune: int = 0
    defaultIndexRatio: int = 1
    defaultProtect: float = 0.5
    isONNX: bool = False
    modelType: str = EnumInferenceTypes.pyTorchRVC.value
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    embOutputLayer: int = 9
    useFinalProj: bool = True
    deprecated: bool = False
    embedder: str = EnumEmbedderTypes.hubert.value

    name: str = ""
    description: str = ""
    credit: str = ""
    termsOfUseUrl: str = ""
    sampleId: str = ""
    iconFile: str = ""
