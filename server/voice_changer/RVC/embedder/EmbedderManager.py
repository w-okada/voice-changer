from torch import device

from const import EnumEmbedderTypes
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.FairseqContentvec import FairseqContentvec
from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert
from voice_changer.RVC.embedder.FairseqHubertJp import FairseqHubertJp


class EmbedderManager:
    currentEmbedder: Embedder | None = None

    @classmethod
    def getEmbedder(
        cls, embederType: EnumEmbedderTypes, file: str, isHalf: bool, dev: device
    ) -> Embedder:
        if cls.currentEmbedder is None:
            print("[Voice Changer] generate new embedder. (no embedder)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, file, isHalf, dev)
        elif cls.currentEmbedder.matchCondition(embederType, file) is False:
            print("[Voice Changer] generate new embedder. (not match)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, file, isHalf, dev)
        else:
            cls.currentEmbedder.setDevice(dev)
            cls.currentEmbedder.setHalf(isHalf)
        return cls.currentEmbedder

    @classmethod
    def loadEmbedder(
        cls, embederType: EnumEmbedderTypes, file: str, isHalf: bool, dev: device
    ) -> Embedder:
        if (
            embederType == EnumEmbedderTypes.hubert
            or embederType == EnumEmbedderTypes.hubert.value
        ):
            return FairseqHubert().loadModel(file, dev, isHalf)
        elif (
            embederType == EnumEmbedderTypes.hubert_jp
            or embederType == EnumEmbedderTypes.hubert_jp.value
        ):
            return FairseqHubertJp().loadModel(file, dev, isHalf)
        elif (
            embederType == EnumEmbedderTypes.contentvec
            or embederType == EnumEmbedderTypes.contentvec.value
        ):
            return FairseqContentvec().loadModel(file, dev, isHalf)
        else:
            return FairseqHubert().loadModel(file, dev, isHalf)
