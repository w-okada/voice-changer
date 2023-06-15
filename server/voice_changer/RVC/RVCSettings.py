from dataclasses import dataclass, field


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
