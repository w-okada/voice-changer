from dataclasses import dataclass, field

from voice_changer.RVC.ModelSlot import ModelSlot


@dataclass
class RVCSettings:
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "harvest"  # pm or harvest
    tran: int = 20
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32
    clusterInferRatio: float = 0.1

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""
    modelSlots: list[ModelSlot] = field(
        default_factory=lambda: [ModelSlot(), ModelSlot(), ModelSlot(), ModelSlot()]
    )
    indexRatio: float = 0
    rvcQuality: int = 0
    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 48000
    modelSlotIndex: int = -1

    speakers: dict[str, int] = field(default_factory=lambda: {})

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
    ]
    floatData = ["silentThreshold", "indexRatio"]
    strData = ["framework", "f0Detector"]
