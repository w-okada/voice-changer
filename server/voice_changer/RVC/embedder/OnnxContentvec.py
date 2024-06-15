import torch
import onnx
import os
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.Embedder import Embedder
import onnxruntime
import numpy as np
from onnxconverter_common import float16


class OnnxContentvec(Embedder):

    def load_model(self, file: str) -> Embedder:
        device_manager = DeviceManager.get_instance()
        self.isHalf = device_manager.use_fp16()
        (
            onnxProviders,
            onnxProviderOptions,
        ) = device_manager.get_onnx_execution_provider()

        if self.isHalf:
            fname, _ = os.path.splitext(os.path.basename(file))
            fp16_fpath = os.path.join(os.path.dirname(file), f'{fname}.fp16.onnx')
            if not os.path.exists(fp16_fpath):
                model: onnx.ModelProto = float16.convert_float_to_float16(onnx.load(file))
                onnx.save(model, fp16_fpath)
            else:
                model = onnx.load(fp16_fpath)
        else:
            model = onnx.load(file)

        so = onnxruntime.SessionOptions()
        # so.log_severity_level = 3
        # so.enable_profiling = True
        self.fp_dtype_t = torch.float16 if self.isHalf else torch.float32
        self.fp_dtype_np = np.float16 if self.isHalf else np.float32
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
