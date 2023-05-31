import torch
import onnxruntime
from const import EnumInferenceTypes
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np


class OnnxRVCInferencer(Inferencer):
    def loadModel(self, file: str, gpu: int):
        self.setProps(EnumInferenceTypes.onnxRVC, file, True, gpu)
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)

        onnx_session = onnxruntime.InferenceSession(
            file, providers=onnxProviders, provider_options=onnxProviderOptions
        )

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

    def getInferencerInfo(self):
        inferencer = super().getInferencerInfo()
        inferencer["onnxExecutionProvider"] = self.model.get_providers()
        return inferencer
