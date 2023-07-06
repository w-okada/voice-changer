from torch import device
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert


class FairseqHubertJp(FairseqHubert):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().loadModel(file, dev, isHalf)
        super().setProps("hubert-base-japanese", file, dev, isHalf)
        return self
