from dataclasses import dataclass, field
from typing import TypeAlias, Union, Any
from const import VoiceChangerType


@dataclass
class ModelSample:
    id: str = ""
    voiceChangerType: VoiceChangerType | None = None


@dataclass
class RVCModelSample(ModelSample):
    id: str = ""
    voiceChangerType: VoiceChangerType = "RVC"
    lang: str = ""
    tag: list[str] = field(default_factory=lambda: [])
    name: str = ""
    modelUrl: str = ""
    indexUrl: str = ""
    termsOfUseUrl: str = ""
    icon: str = ""
    credit: str = ""
    description: str = ""

    sampleRate: int = 48000
    modelType: str = ""
    f0: bool = True


ModelSamples: TypeAlias = Union[ModelSample, RVCModelSample]


def generateModelSample(params: Any) -> ModelSamples:
    if params["voiceChangerType"] == "RVC":
        return RVCModelSample(**{k: v for k, v in params.items() if k in RVCModelSample.__annotations__})
    else:
        return ModelSample(**{k: v for k, v in params.items() if k in ModelSample.__annotations__})
