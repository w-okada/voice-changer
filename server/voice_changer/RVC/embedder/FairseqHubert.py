import torch
from torch import device
from const import EnumEmbedderTypes
from voice_changer.RVC.embedder.Embedder import Embedder
from fairseq import checkpoint_utils


class FairseqHubert(Embedder):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().setProps(EnumEmbedderTypes.hubert, file, dev, isHalf)

        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [file],
            suffix="",
        )
        model = models[0]
        model.eval()

        model = model.to(dev)
        if isHalf:
            model = model.half()

        self.model = model
        return self

    def extractFeatures(self, feats: torch.Tensor, embChannels=256) -> torch.Tensor:
        padding_mask = torch.BoolTensor(feats.shape).to(self.dev).fill_(False)
        if embChannels == 256:
            inputs = {
                "source": feats.to(self.dev),
                "padding_mask": padding_mask,
                "output_layer": 9,  # layer 9
            }
        else:
            inputs = {
                "source": feats.to(self.dev),
                "padding_mask": padding_mask,
            }

        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            if embChannels == 256:
                feats = self.model.final_proj(logits[0])
            else:
                feats = logits[0]
        return feats
