import torch
import onnxruntime
from const import EnumInferenceTypes
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np
import os
import onnx
from onnxconverter_common import float16


class OnnxRVCInferencer(Inferencer):
    def loadModel(self, file: str, gpu: int, inferencerTypeVersion: str | None = None):
        self.inferencerTypeVersion = inferencerTypeVersion
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)
        # FIXME: Temporarily disable conversion to FP16 since the forward pass contains the ops that are not converted correctly.
        # Note that float model inputs are explicitly converted to FP32!
        # self.isHalf = DeviceManager.get_instance().halfPrecisionAvailable(gpu)
        self.isHalf = False

        self.setProps(EnumInferenceTypes.onnxRVC, file, self.isHalf, gpu)

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

        self.fp_dtype_t = torch.float16 if self.isHalf else torch.float32
        self.fp_dtype_np = np.float16 if self.isHalf else np.float32

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
        skip_head: torch.Tensor | None,
        return_length: torch.Tensor | None,
    ) -> torch.Tensor:
        if pitch is None or pitchf is None:
            raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")

        if feats.device.type == 'cuda':
            binding = self.model.io_binding()

            binding.bind_input('feats', device_type='cuda', device_id=feats.device.index, element_type=self.fp_dtype_np, shape=tuple(feats.shape), buffer_ptr=feats.float().data_ptr())
            binding.bind_input('p_len', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(pitch_length.shape), buffer_ptr=pitch_length.data_ptr())
            binding.bind_input('pitch', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(pitch.shape), buffer_ptr=pitch.data_ptr())
            binding.bind_input('pitchf', device_type='cuda', device_id=feats.device.index, element_type=self.fp_dtype_np, shape=tuple(pitchf.shape), buffer_ptr=pitchf.float().data_ptr())
            binding.bind_input('sid', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(sid.shape), buffer_ptr=sid.data_ptr())
            binding.bind_cpu_input('skip_head', np.array(skip_head, dtype=np.int64))

            binding.bind_output('audio', device_type='cuda', device_id=feats.device.index)

            self.model.run_with_iobinding(binding)

            output = [output.numpy() for output in binding.get_outputs()]
        else:
            output = self.model.run(
                ["audio"],
                {
                    "feats": feats.float().detach().cpu().numpy(),
                    "p_len": pitch_length.detach().cpu().numpy(),
                    "pitch": pitch.detach().cpu().numpy(),
                    "pitchf": pitchf.float().detach().cpu().numpy(),
                    "sid": sid.detach().cpu().numpy(),
                    "skip_head": np.array(skip_head, dtype=np.int64)
                },
            )
        # self.model.end_profiling()

        res = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=sid.device)
        if self.isHalf:
            res = res.float()

        if self.inferencerTypeVersion == "v2.1" or self.inferencerTypeVersion == "v2.2" or self.inferencerTypeVersion == "v1.1":
            return res
        return torch.clip(res[0, 0], -1.0, 1.0)

    def getInferencerInfo(self):
        inferencer = super().getInferencerInfo()
        inferencer["onnxExecutionProvider"] = self.model.get_providers()
        return inferencer
