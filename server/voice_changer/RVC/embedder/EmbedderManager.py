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
            cls.loadEmbedder(embederType, file, isHalf, dev)
            cls.currentEmbedder = cls.loadEmbedder(embederType, file, isHalf, dev)
        elif cls.currentEmbedder.matchCondition(embederType, file) is False:
            print("[Voice Changer] generate new embedder. (not match)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, file, isHalf, dev)
        else:
            cls.currentEmbedder.setDevice(dev)
            cls.currentEmbedder.setHalf(isHalf)
        print("RETURN", cls.currentEmbedder)
        return cls.currentEmbedder

    @classmethod
    def loadEmbedder(
        cls, embederType: EnumEmbedderTypes, file: str, isHalf: bool, dev: device
    ) -> Embedder:
        if embederType == EnumEmbedderTypes.hubert:
            return FairseqHubert().loadModel(file, dev, isHalf)
        elif embederType == EnumEmbedderTypes.hubert_jp:  # same as hubert
            return FairseqHubertJp().loadModel(file, dev, isHalf)
        elif embederType == EnumEmbedderTypes.contentvec:  # same as hubert
            return FairseqContentvec().loadModel(file, dev, isHalf)
        else:
            # return hubert as default
            return FairseqHubert().loadModel(file, dev, isHalf)
