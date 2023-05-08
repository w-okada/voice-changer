from dataclasses import dataclass


@dataclass
class ModelSlot:
    modelFile: str = ""
    diffusionFile: str = ""
    defaultTrans: int = 0
