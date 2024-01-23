from torch import device

from const import EmbedderType
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.OnnxContentvec import OnnxContentvec
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class EmbedderManager:
    currentEmbedder: Embedder | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getEmbedder(
        cls, embederType: EmbedderType, isHalf: bool, dev: device
    ) -> Embedder:
        if cls.currentEmbedder is None:
            print("[Voice Changer] generate new embedder. (no embedder)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        elif cls.currentEmbedder.matchCondition(embederType) is False:
            print("[Voice Changer] generate new embedder. (not match)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        else:
            print("[Voice Changer] generate new embedder. (anyway)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)

            # cls.currentEmbedder.setDevice(dev)
            # cls.currentEmbedder.setHalf(isHalf)
        return cls.currentEmbedder

    @classmethod
    def loadEmbedder(
        cls, embederType: EmbedderType, isHalf: bool, dev: device
    ) -> Embedder:
        if embederType not in ["hubert_base", "contentvec"]:
            raise RuntimeError(f'Unsupported embedder type: {embederType}')
        file = cls.params.content_vec_500_onnx
        return OnnxContentvec().loadModel(file, dev)
