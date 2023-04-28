from dataclasses import dataclass
from voice_changer.RVC.const import RVC_MODEL_TYPE_RVC


@dataclass
class ModelSlot:
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    featureFile: str = ""
    indexFile: str = ""
    defaultTrans: int = 0
    modelType: int = RVC_MODEL_TYPE_RVC
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    deprecated: bool = False
    embedder: str = "hubert_base"  # "hubert_base",  "contentvec",  "distilhubert"
