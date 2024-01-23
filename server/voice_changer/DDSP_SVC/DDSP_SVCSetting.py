from dataclasses import dataclass, field


@dataclass
class DDSP_SVCSettings:
    gpu: int = -9999
    dstId: int = 1

    f0Detector: str = "dio"  # dio or harvest or crepe
    tran: int = 20
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32

    useEnhancer: int = 0
    useDiff: int = 1
    # useDiffDpm: int = 0
    diffMethod: str = "dpm-solver"  # "pndm", "dpm-solver"
    useDiffSilence: int = 0
    diffAcc: int = 20
    diffSpkId: int = 1
    kStep: int = 120
    threshold: int = -45

    speakers: dict[str, int] = field(default_factory=lambda: {})
    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "useEnhancer",
        "useDiff",
        # "useDiffDpm",
        "useDiffSilence",
        "diffAcc",
        "diffSpkId",
        "kStep",
    ]
    floatData = ["silentThreshold"]
    strData = ["f0Detector", "diffMethod"]
