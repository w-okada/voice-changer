from dataclasses import dataclass, field
from ModelSample import RVCModelSample
from const import MAX_SLOT_NUM
from data.ModelSlot import ModelSlot, ModelSlots


@dataclass
class RVCSettings:
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "harvest"  # dio or harvest
    tran: int = 20
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32
    clusterInferRatio: float = 0.1

    framework: str = "PyTorch"  # PyTorch or ONNX
    modelSlots: list[ModelSlots] = field(default_factory=lambda: [ModelSlot() for _x in range(MAX_SLOT_NUM)])

    sampleModels: list[RVCModelSample] = field(default_factory=lambda: [])

    indexRatio: float = 0
    protect: float = 0.5
    rvcQuality: int = 0
    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 48000
    modelSlotIndex: int = -1

    speakers: dict[str, int] = field(default_factory=lambda: {})
    isHalf: int = 1  # 0:off, 1:on
    enableDirectML: int = 0  # 0:off, 1:on
    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "rvcQuality",
        "modelSamplingRate",
        "silenceFront",
        "modelSlotIndex",
        "isHalf",
        "enableDirectML",
    ]
    floatData = ["silentThreshold", "indexRatio", "protect"]
    strData = ["framework", "f0Detector"]
