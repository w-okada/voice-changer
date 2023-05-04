import torch
from torch import device
import onnxruntime
from const import EnumInferenceTypes
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np

providers = ["CPUExecutionProvider"]


class OnnxRVCInferencer(Inferencer):
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
        self.setDevice(dev)
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
    ) -> torch.Tensor:
        if pitch is None or pitchf is None:
            raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")

        # print("INFER1", self.model.get_providers())
        # print("INFER2", self.model.get_provider_options())
        # print("INFER3", self.model.get_session_options())

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
        self.isHalf = isHalf
        pass
        # raise RuntimeError("half-precision is not changable.", self.isHalf)

    def setDevice(self, dev: device):
        index = dev.index
        type = dev.type
        if type == "cpu":
            self.model.set_providers(providers=["CPUExecutionProvider"])
        elif type == "cuda":
            provider_options = [{"device_id": index}]
            self.model.set_providers(
                providers=["CUDAExecutionProvider"],
                provider_options=provider_options,
            )
        else:
            self.model.set_providers(providers=["CPUExecutionProvider"])

        return self

    def setDirectMLEnable(self, enable: bool):
        if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            print("[Voice Changer] DML is not available.")
            return

        if enable:
            self.model.set_providers(
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]
            )
        else:
            self.model.set_providers(providers=["CPUExecutionProvider"])
