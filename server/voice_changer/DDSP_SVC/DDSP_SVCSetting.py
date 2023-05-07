from dataclasses import dataclass, field


@dataclass
class DDSP_SVCSettings:
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "dio"  # dio or harvest # parselmouth
    tran: int = 20
    predictF0: int = 0  # 0:False, 1:True
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32

    enableEnhancer: int = 0
    enhancerTune: int = 0

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    speakers: dict[str, int] = field(default_factory=lambda: {})

    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "predictF0",
        "extraConvertSize",
        "enableEnhancer",
        "enhancerTune",
    ]
    floatData = ["silentThreshold", "clusterInferRatio"]
    strData = ["framework", "f0Detector"]
