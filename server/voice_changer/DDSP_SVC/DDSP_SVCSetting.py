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

    enableEnhancer: int = 0
    enhancerTune: int = 0

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
        "enableEnhancer",
        "enhancerTune",
    ]
    floatData = ["silentThreshold"]
    strData = ["framework", "f0Detector"]
