import torch
from torch import device
import onnxruntime
from const import EnumInferenceTypes
import numpy as np

from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer

providers = ["CPUExecutionProvider"]


class OnnxRVCInferencerNono(OnnxRVCInferencer):
    def loadModel(self, file: str, dev: device, isHalf: bool = True):
        super().setProps(EnumInferenceTypes.onnxRVC, file, dev, isHalf)
        # ort_options = onnxruntime.SessionOptions()
        # ort_options.intra_op_num_threads = 8

        onnx_session = onnxruntime.InferenceSession(file, providers=providers)

        # check half-precision
        first_input_type = onnx_session.get_inputs()[0].type
        if first_input_type == "tensor(float)":
            self.isHalf = False
        else:
            self.isHalf = True

        self.model = onnx_session
        return self

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
