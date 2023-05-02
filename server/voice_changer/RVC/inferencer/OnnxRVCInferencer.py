import torch
from torch import device
import onnxruntime
from const import EnumInferenceTypes
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np

providers = ["CPUExecutionProvider"]


class OnnxRVCInference(Inferencer):
    def loadModel(self, file: str, dev: device, isHalf: bool = True):
        super().setProps(EnumInferenceTypes.onnxRVC, file, dev, isHalf)
        # ort_options = onnxruntime.SessionOptions()
        # ort_options.intra_op_num_threads = 8

        onnx_session = onnxruntime.InferenceSession(
            self.onnx_model, providers=providers
        )

        # check half-precision
        first_input_type = self.onnx_session.get_inputs()[0].type
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
        if pitch is None or pitchf is None:
            raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")

        if self.isHalf:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float16),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                    "pitch": pitch.cpu().numpy().astype(np.int64),
                    "pitchf": pitchf.cpu().numpy().astype(np.float32),
                    "sid": sid.cpu().numpy().astype(np.int64),
                },
            )
        else:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float32),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                    "pitch": pitch.cpu().numpy().astype(np.int64),
                    "pitchf": pitchf.cpu().numpy().astype(np.float32),
                    "sid": sid.cpu().numpy().astype(np.int64),
                },
            )

        return torch.tensor(np.array(audio1))

    def setHalf(self, isHalf: bool):
        raise RuntimeError("half-precision is not changable.", self.isHalf)

    def setDevice(self, dev: device):
        self.dev = dev
        if self.model is not None:
            self.model = self.model.to(self.dev)

    def to(self, dev: torch.device):
        if self.model is not None:
            self.model = self.model.to(dev)
        return self
