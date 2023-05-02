from torch import device
from const import EnumEmbedderTypes
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert


class FairseqHubertJp(FairseqHubert):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().loadModel(file, dev, isHalf)
        super().setProps(EnumEmbedderTypes.hubert_jp, file, dev, isHalf)
        return self
