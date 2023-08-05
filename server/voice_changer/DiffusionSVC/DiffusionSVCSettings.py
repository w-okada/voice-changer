from dataclasses import dataclass, field


@dataclass
class DiffusionSVCSettings:
    gpu: int = -9999
    dstId: int = 0

    f0Detector: str = "harvest"  # dio or harvest
    tran: int = 12
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 4

    kStep: int = 20
    speedUp: int = 10
    skipDiffusion: int = 0  # 0:off, 1:on

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
        "kStep",
        "speedUp",
        "silenceFront",
    ]
    floatData = ["silentThreshold"]
    strData = ["f0Detector"]
