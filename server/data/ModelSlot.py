from typing import TypeAlias, Union
from const import MAX_SLOT_NUM, DiffusionSVCInferenceType, EnumInferenceTypes, EmbedderType, VoiceChangerType

from dataclasses import dataclass, asdict, field

import os
import json


@dataclass
class ModelSlot:
    slotIndex: int = -1
    voiceChangerType: VoiceChangerType | None = None
    name: str = ""
    description: str = ""
    credit: str = ""
    termsOfUseUrl: str = ""
    iconFile: str = ""
    speakers: dict = field(default_factory=lambda: {})


@dataclass
class RVCModelSlot(ModelSlot):
    voiceChangerType: VoiceChangerType = "RVC"
    modelFile: str = ""
    indexFile: str = ""
    defaultTune: int = 0
    defaultIndexRatio: int = 0
    defaultProtect: float = 0.5
    isONNX: bool = False
    modelType: str = EnumInferenceTypes.pyTorchRVC.value
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    embOutputLayer: int = 9
    useFinalProj: bool = True
    deprecated: bool = False
    embedder: EmbedderType = "hubert_base"

    sampleId: str = ""
    speakers: dict = field(default_factory=lambda: {0: "target"})


@dataclass
class MMVCv13ModelSlot(ModelSlot):
    voiceChangerType: VoiceChangerType = "MMVCv13"
    modelFile: str = ""
    configFile: str = ""
    srcId: int = 107
    dstId: int = 100
    isONNX: bool = False
    samplingRate: int = 24000
    speakers: dict = field(default_factory=lambda: {107: "user", 100: "zundamon", 101: "sora", 102: "methane", 103: "tsumugi"})


@dataclass
class MMVCv15ModelSlot(ModelSlot):
    voiceChangerType: VoiceChangerType = "MMVCv15"
    modelFile: str = ""
    configFile: str = ""
    srcId: int = 0
    dstId: int = 101
    f0Factor: float = 1.0
    isONNX: bool = False
    samplingRate: int = 24000
    speakers: dict = field(default_factory=lambda: {})
    f0: dict = field(default_factory=lambda: {})


@dataclass
class SoVitsSvc40ModelSlot(ModelSlot):
    voiceChangerType: VoiceChangerType = "so-vits-svc-40"
    modelFile: str = ""
    configFile: str = ""
    clusterFile: str = ""
    dstId: int = 0
    isONNX: bool = False

    sampleId: str = ""

    defaultTune: int = 0
    defaultClusterInferRatio: float = 0.0
    noiseScale: float = 0.0
    speakers: dict = field(default_factory=lambda: {1: "user"})


@dataclass
class DDSPSVCModelSlot(ModelSlot):
    voiceChangerType: VoiceChangerType = "DDSP-SVC"
    modelFile: str = ""
    configFile: str = ""
    diffModelFile: str = ""
    diffConfigFile: str = ""
    dstId: int = 0
    isONNX: bool = False

    sampleId: str = ""
    defaultTune: int = 0
    enhancer: bool = False
    diffusion: bool = True
    acc: int = 20
    kstep: int = 100
    speakers: dict = field(default_factory=lambda: {1: "user"})


@dataclass
class DiffusionSVCModelSlot(ModelSlot):
    voiceChangerType: VoiceChangerType = "Diffusion-SVC"
    modelFile: str = ""
    isONNX: bool = False
    modelType: DiffusionSVCInferenceType = "combo"
    dstId: int = 1

    sampleId: str = ""
    defaultTune: int = 0
    defaultKstep: int = 20
    defaultSpeedup: int = 10
    kStepMax: int = 100
    nLayers: int = 20
    nnLayers: int = 20
    speakers: dict = field(default_factory=lambda: {1: "user"})
    embedder: EmbedderType = "hubert_base"
    samplingRate: int = 44100
    embChannels: int = 768


ModelSlots: TypeAlias = Union[ModelSlot, RVCModelSlot, MMVCv13ModelSlot, MMVCv15ModelSlot, SoVitsSvc40ModelSlot, DDSPSVCModelSlot, DiffusionSVCModelSlot]


def loadSlotInfo(model_dir: str, slotIndex: int) -> ModelSlots:
    slotDir = os.path.join(model_dir, str(slotIndex))
    jsonFile = os.path.join(slotDir, "params.json")
    if not os.path.exists(jsonFile):
        return ModelSlot()
    jsonDict = json.load(open(os.path.join(slotDir, "params.json")))
    slotInfoKey = list(ModelSlot.__annotations__.keys())
    slotInfo = ModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    if slotInfo.voiceChangerType == "RVC":
        slotInfoKey.extend(list(RVCModelSlot.__annotations__.keys()))
        return RVCModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    elif slotInfo.voiceChangerType == "MMVCv13":
        slotInfoKey.extend(list(MMVCv13ModelSlot.__annotations__.keys()))
        return MMVCv13ModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    elif slotInfo.voiceChangerType == "MMVCv15":
        slotInfoKey.extend(list(MMVCv15ModelSlot.__annotations__.keys()))
        return MMVCv15ModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    elif slotInfo.voiceChangerType == "so-vits-svc-40":
        slotInfoKey.extend(list(SoVitsSvc40ModelSlot.__annotations__.keys()))
        return SoVitsSvc40ModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    elif slotInfo.voiceChangerType == "DDSP-SVC":
        slotInfoKey.extend(list(DDSPSVCModelSlot.__annotations__.keys()))
        return DDSPSVCModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    elif slotInfo.voiceChangerType == "Diffusion-SVC":
        slotInfoKey.extend(list(DiffusionSVCModelSlot.__annotations__.keys()))
        return DiffusionSVCModelSlot(**{k: v for k, v in jsonDict.items() if k in slotInfoKey})
    else:
        return ModelSlot()


def loadAllSlotInfo(model_dir: str):
    slotInfos: list[ModelSlots] = []
    for slotIndex in range(MAX_SLOT_NUM):
        slotInfo = loadSlotInfo(model_dir, slotIndex)
        slotInfo.slotIndex = slotIndex  # スロットインデックスは動的に注入
        slotInfos.append(slotInfo)
    return slotInfos


def saveSlotInfo(model_dir: str, slotIndex: int, slotInfo: ModelSlots):
    slotDir = os.path.join(model_dir, str(slotIndex))
    slotInfoDict = asdict(slotInfo)
    slotInfo.slotIndex = -1  # スロットインデックスは動的に注入
    json.dump(slotInfoDict, open(os.path.join(slotDir, "params.json"), "w"), indent=4)
