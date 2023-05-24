from const import EnumInferenceTypes, EnumEmbedderTypes

from dataclasses import dataclass


@dataclass
class ModelSlot:
    # pyTorchModelFile: str = ""
    # onnxModelFile: str = ""
    modelFile: str = ""
    featureFile: str = ""
    indexFile: str = ""
    defaultTune: int = 0
    defaultIndexRatio: int = 1
    isONNX: bool = False
    modelType: EnumInferenceTypes = EnumInferenceTypes.pyTorchRVC
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    embOutputLayter: int = 9
    useFinalProj: bool = True
    deprecated: bool = False
    embedder: EnumEmbedderTypes = EnumEmbedderTypes.hubert

    name: str = ""
    description: str = ""
    credit: str = ""
    termsOfUseUrl: str = ""
