from dataclasses import dataclass, field


@dataclass
class BeatriceSettings:
    # gpu: int = -9999
    dstId: int = 0
    modelSamplingRate: int = 48000
    silentThreshold: float = 0.00001
    speakers: dict[str, int] = field(default_factory=lambda: {})
    intData = [
        # "gpu",
        "dstId",
    ]
    floatData = ["silentThreshold"]
    strData = []
