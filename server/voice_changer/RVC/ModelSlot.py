from const import EnumInferenceTypes, EnumEmbedderTypes

from dataclasses import dataclass


@dataclass
class ModelSlot:
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    featureFile: str = ""
    indexFile: str = ""
    defaultTrans: int = 0
    isONNX: bool = False
    modelType: EnumInferenceTypes = EnumInferenceTypes.pyTorchRVC
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    deprecated: bool = False
    embedder: EnumEmbedderTypes = EnumEmbedderTypes.hubert
