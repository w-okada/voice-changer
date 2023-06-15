from dataclasses import dataclass

from voice_changer.VoiceChanger import SlotInfo


@dataclass
class RVCSlotInfo(SlotInfo):
    modelFile: str = ""
    indexFile: str = ""
    defaultTune: int = 0
    defaultIndexRatio: float = 0
    defaultProtect: float = 1
    isONNX: bool = False
    modelType: str = ""
    samplingRate: int = 40000
    f0: bool = True
    embChannels: int = 256
    embOutputLayer: int = 12
    useFinalProj: bool = False
    deprecated: bool = False
    embedder: str = ""

    name: str = ""
    description: str = ""
    credit: str = ""
    termsOfUseUrl: str = ""
    sampleId: str = ""
    iconFile: str = ""
