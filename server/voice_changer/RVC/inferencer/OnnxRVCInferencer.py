import torch
import onnxruntime
from const import EnumInferenceTypes
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np


class OnnxRVCInferencer(Inferencer):
    def loadModel(self, file: str, gpu: int, inferencerTypeVersion: str | None = None):
        self.setProps(EnumInferenceTypes.onnxRVC, file, True, gpu)
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        onnx_session = onnxruntime.InferenceSession(
            file, sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions
        )

        # check half-precision of "feats" input
        self.input_feats_half = onnx_session.get_inputs()[0].type == "tensor(float16)"
        self.model = onnx_session

        # self.output_half = onnx_session.get_outputs()[0].type == "tensor(float16)"

        self.inferencerTypeVersion = inferencerTypeVersion

        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        convert_length: int | None,
    ) -> torch.Tensor:
        if pitch is None or pitchf is None:
            raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")

        # print("INFER1", self.model.get_providers())
        # print("INFER2", self.model.get_provider_options())
        # print("INFER3", self.model.get_session_options())
        if feats.device.type == 'cuda':
            binding = self.model.io_binding()

            if self.input_feats_half:
                feats = feats.to(torch.float16)
                binding.bind_input('feats', device_type='cuda', device_id=feats.device.index, element_type=np.float16, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
            else:
                binding.bind_input('feats', device_type='cuda', device_id=feats.device.index, element_type=np.float32, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
            binding.bind_input('p_len', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(pitch_length.shape), buffer_ptr=pitch_length.data_ptr())
            binding.bind_input('pitch', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(pitch.shape), buffer_ptr=pitch.data_ptr())
            binding.bind_input('pitchf', device_type='cuda', device_id=feats.device.index, element_type=np.float32, shape=tuple(pitchf.shape), buffer_ptr=pitchf.data_ptr())
            binding.bind_input('sid', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(sid.shape), buffer_ptr=sid.data_ptr())

            binding.bind_output('audio', device_type='cuda', device_id=feats.device.index)

            self.model.run_with_iobinding(binding)

            output = [output.numpy() for output in binding.get_outputs()]
        else:
            output = self.model.run(
                ["audio"],
                {
                    "feats": feats.detach().cpu().numpy().astype(np.float16 if self.input_feats_half else np.float32, copy=False),
                    "p_len": pitch_length.detach().cpu().numpy(),
                    "pitch": pitch.detach().cpu().numpy(),
                    "pitchf": pitchf.detach().cpu().numpy(),
                    "sid": sid.detach().cpu().numpy()
                },
            )
        # self.model.end_profiling()

        res = torch.as_tensor(output[0], dtype=torch.float32, device=sid.device)

        if self.inferencerTypeVersion == "v2.1" or self.inferencerTypeVersion == "v2.2" or self.inferencerTypeVersion == "v1.1":
            return res
        return torch.clip(res[0, 0], -1.0, 1.0)

    def getInferencerInfo(self):
        inferencer = super().getInferencerInfo()
        inferencer["onnxExecutionProvider"] = self.model.get_providers()
        return inferencer
