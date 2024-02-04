import torch
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.Embedder import Embedder
import onnxruntime
import numpy as np


class OnnxContentvec(Embedder):

    def loadModel(self, file: str, dev: torch.device) -> Embedder:
        gpu = dev.index if dev.index is not None else -1
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True

        self.onnx_session = onnxruntime.InferenceSession(file, sess_options=so, providers=onnxProviders, provider_options=onnxProviderOptions)
        super().setProps('hubert_base', file, dev, False)
        return self

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        if feats.device.type == 'cuda':
            binding = self.onnx_session.io_binding()

            binding.bind_input('audio', device_type='cuda', device_id=feats.device.index, element_type=np.float32, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
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
            dtype=torch.float32,
            device=feats.device
        )
