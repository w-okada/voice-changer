from dataclasses import dataclass, field


@dataclass
class RVCSettings:
    gpu: int = -1
    dstId: int = 0

    f0Detector: str = "rmvpe_onnx"  # dio or harvest
    tran: int = 12
    silentThreshold: float = 0.00001

    indexRatio: float = 0
    protect: float = 0.5
    silenceFront: int = 1  # 0:off, 1:on
    forceFp32: int = 0 # 0:off, 1:on
    modelSamplingRate: int = 48000

    speakers: dict[str, int] = field(default_factory=lambda: {})
    # isHalf: int = 1  # 0:off, 1:on
    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "forceFp32",
        "dstId",
        "tran",
        "extraConvertSize",
        "silenceFront",
    ]
    floatData = ["silentThreshold", "indexRatio", "protect"]
    strData = ["f0Detector"]
