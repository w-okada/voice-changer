import torch
import onnxruntime
from const import EnumInferenceTypes
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np

class OnnxRVCInferencer(Inferencer):
    def load_model(self, file: str, inferencerTypeVersion: str | None = None):
        self.inferencerTypeVersion = inferencerTypeVersion
        device_manager = DeviceManager.get_instance()
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().get_onnx_execution_provider()
        self.is_half = device_manager.use_fp16()

        self.set_props(EnumInferenceTypes.onnxRVC, file)

        model = load_onnx_model(file, self.is_half)

        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        self.model = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)

        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        skip_head: int,
        return_length: int,
        formant_length: int,
    ) -> torch.Tensor:
        assert pitch is not None or pitchf is not None, "Pitch or Pitchf is not found."

        if feats.device.type == 'cuda':
            binding = self.model.io_binding()

            binding.bind_input('feats', device_type='cuda', device_id=feats.device.index, element_type=self.fp_dtype_np, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
            binding.bind_input('p_len', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(pitch_length.shape), buffer_ptr=pitch_length.data_ptr())
            binding.bind_input('pitch', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(pitch.shape), buffer_ptr=pitch.data_ptr())
            binding.bind_input('pitchf', device_type='cuda', device_id=feats.device.index, element_type=self.fp_dtype_np, shape=tuple(pitchf.shape), buffer_ptr=pitchf.data_ptr())
            binding.bind_input('sid', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(sid.shape), buffer_ptr=sid.data_ptr())
            binding.bind_cpu_input('skip_head', np.array(skip_head, dtype=np.int64))
            binding.bind_cpu_input('return_length', np.array(return_length, dtype=np.int64))
            binding.bind_cpu_input('formant_length', np.array(formant_length, dtype=np.int64))

            binding.bind_output('audio', device_type='cuda', device_id=feats.device.index)

            self.model.run_with_iobinding(binding)

            output = [output.numpy() for output in binding.get_outputs()]
        else:
            output = self.model.run(
                ["audio"],
                {
                    "feats": feats.detach().cpu().numpy(),
                    "p_len": pitch_length.detach().cpu().numpy(),
                    "pitch": pitch.detach().cpu().numpy(),
                    "pitchf": pitchf.detach().cpu().numpy(),
                    "sid": sid.detach().cpu().numpy(),
                    "skip_head": np.array(skip_head, dtype=np.int64),
                    "return_length": np.array(return_length, dtype=np.int64),
                    "formant_length": np.array(formant_length, dtype=np.int64),
                },
            )
        # self.model.end_profiling()

        res = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=feats.device)

        if self.inferencerTypeVersion == "v2.1" or self.inferencerTypeVersion == "v2.2" or self.inferencerTypeVersion == "v1.1":
            return res
        return torch.clip(res[0, 0], -1.0, 1.0)

    def getInferencerInfo(self):
        inferencer = super().getInferencerInfo()
        inferencer["onnxExecutionProvider"] = self.model.get_providers()
        return inferencer
