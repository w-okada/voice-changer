from dataclasses import dataclass, field

from const import PitchExtractorType


@dataclass
class EasyVCSettings:
    gpu: int = -9999
    dstId: int = 0

    f0Detector: PitchExtractorType = "rmvpe_onnx"  # dio or harvest
    tran: int = 12
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 4

    indexRatio: float = 0
    protect: float = 0.5
    rvcQuality: int = 0
    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 48000

    speakers: dict[str, int] = field(default_factory=lambda: {})

    intData = [
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "rvcQuality",
        "silenceFront",
    ]
    floatData = ["silentThreshold", "indexRatio", "protect"]
    strData = ["f0Detector"]
