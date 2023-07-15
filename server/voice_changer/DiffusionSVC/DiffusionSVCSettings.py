from dataclasses import dataclass, field


@dataclass
class DiffusionSVCSettings:
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "harvest"  # dio or harvest
    tran: int = 12
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 4

    kstep: int = 20
    speedup: int = 10

    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 44100

    speakers: dict[str, int] = field(default_factory=lambda: {})
    # isHalf: int = 1  # 0:off, 1:on
    # enableDirectML: int = 0  # 0:off, 1:on
    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "kstep",
        "silenceFront",
    ]
    floatData = ["silentThreshold"]
    strData = ["f0Detector"]
