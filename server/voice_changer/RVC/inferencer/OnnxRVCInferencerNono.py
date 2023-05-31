import torch
import numpy as np
from const import EnumInferenceTypes

from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer


class OnnxRVCInferencerNono(OnnxRVCInferencer):
    def loadModel(self, file: str, gpu: int):
        super().loadModel(file, gpu)
        self.setProps(EnumInferenceTypes.onnxRVCNono, file, True, gpu)

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
    ) -> torch.Tensor:
        if self.isHalf:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float16),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                    "sid": sid.cpu().numpy().astype(np.int64),
                },
            )
        else:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float32),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                    "sid": sid.cpu().numpy().astype(np.int64),
                },
            )

        return torch.tensor(np.array(audio1))
