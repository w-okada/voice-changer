from dataclasses import dataclass, field

from voice_changer.DDSP_SVC.ModelSlot import ModelSlot


@dataclass
class DDSP_SVCSettings:
    gpu: int = 0
    dstId: int = 1

    f0Detector: str = "dio"  # dio or harvest or crepe # parselmouth
    tran: int = 20
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32

    useEnhancer: int = 0
    useDiff: int = 1
    useDiffDpm: int = 0
    useDiffSilence: int = 0
    diffAcc: int = 20
    diffSpkId: int = 1
    kStep: int = 120
    threshold: int = -45

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    speakers: dict[str, int] = field(default_factory=lambda: {})
    modelSlotIndex: int = -1
    modelSlots: list[ModelSlot] = field(default_factory=lambda: [ModelSlot()])
    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "extraConvertSize",
        "useEnhancer",
        "useDiff",
        "useDiffDpm",
        "useDiffSilence",
        "diffAcc",
        "diffSpkId",
        "kStep",
    ]
    floatData = ["silentThreshold"]
    strData = ["framework", "f0Detector"]
