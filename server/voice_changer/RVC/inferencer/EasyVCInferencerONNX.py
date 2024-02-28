import torch
import numpy as np
from const import EnumInferenceTypes

from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer


class EasyVCInferencerONNX(OnnxRVCInferencer):
    def loadModel(self, file: str, gpu: int, inferencerTypeVersion: str | None = None):
        super().loadModel(file, gpu, inferencerTypeVersion)
        self.setProps(EnumInferenceTypes.easyVC, file, self.isHalf, gpu)
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        convert_length: int | None,
    ) -> torch.Tensor:
        if self.isHalf:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float16),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                },
            )
        else:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float32),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                },
            )
        res = audio1[0][0][0]

        # if self.inferencerTypeVersion == "v2.1" or self.inferencerTypeVersion == "v1.1":
        #     res = audio1[0]
        # else:
        #     res = np.array(audio1)[0][0, 0]
        #     res = np.clip(res, -1.0, 1.0)
        return torch.tensor(res)
