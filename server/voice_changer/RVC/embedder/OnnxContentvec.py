import torch
from voice_changer.common.OnnxLoader import load_onnx_model
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.Embedder import Embedder
import onnxruntime
import numpy as np

class OnnxContentvec(Embedder):

    def load_model(self, file: str) -> Embedder:
        device_manager = DeviceManager.get_instance()
        self.is_half = device_manager.use_fp16()
        (
            onnxProviders,
            onnxProviderOptions,
        ) = device_manager.get_onnx_execution_provider()

        model = load_onnx_model(file, self.is_half)

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        self.fp_dtype_t = torch.float16 if self.is_half else torch.float32
        self.fp_dtype_np = np.float16 if self.is_half else np.float32
        self.onnx_session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)
        super().set_props('hubert_base', file)
        return self

    def extract_features(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        if feats.device.type == 'cuda':
            binding = self.onnx_session.io_binding()

            binding.bind_input('audio', device_type='cuda', device_id=feats.device.index, element_type=self.fp_dtype_np, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
            for output in self.onnx_session.get_outputs():
                binding.bind_output(output.name, device_type='cuda', device_id=feats.device.index)

            self.onnx_session.run_with_iobinding(binding)

            units = [output.numpy() for output in binding.get_outputs()]
        else:
            units = self.onnx_session.run(
                ['units9', 'unit12', 'unit12s'],
                { 'audio': feats.detach().cpu().numpy() }
            )
        # self.onnx_session.end_profiling()

        return torch.as_tensor(
            units[0] if embOutputLayer == 9 else units[1],
            dtype=self.fp_dtype_t,
            device=feats.device
        )
