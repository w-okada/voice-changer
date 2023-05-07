from const import EnumInferenceTypes, EnumEmbedderTypes

from dataclasses import dataclass


@dataclass
class ModelSlot:
    pyTorchModelFile: str = ""
    pyTorchDiffusionModelFile: str = ""
    defaultTrans: int = 0
    # modelType: EnumDDSPSVCInferenceTypes = EnumDDSPSVCInferenceTypes.pyTorchRVC
    # samplingRate: int = -1
    # f0: bool = True
    # embChannels: int = 256
    # deprecated: bool = False
    # embedder: EnumEmbedderTypes = EnumEmbedderTypes.hubert
