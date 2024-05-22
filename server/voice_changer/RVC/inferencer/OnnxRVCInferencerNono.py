import torch
import numpy as np
from const import EnumInferenceTypes

from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer


class OnnxRVCInferencerNono(OnnxRVCInferencer):
    def loadModel(self, file: str, gpu: int, inferencerTypeVersion: str | None = None):
        super().loadModel(file, gpu, inferencerTypeVersion)
        self.setProps(EnumInferenceTypes.onnxRVCNono, file, self.isHalf, gpu)
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        skip_head: torch.Tensor | None,
        return_length: torch.Tensor | None,
    ) -> torch.Tensor:
        if feats.device.type == 'cuda':
            binding = self.model.io_binding()

            if self.input_feats_half:
                feats = feats.to(torch.float16)
                binding.bind_input('feats', device_type='cuda', device_id=feats.device.index, element_type=np.float16, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
            else:
                binding.bind_input('feats', device_type='cuda', device_id=feats.device.index, element_type=np.float32, shape=tuple(feats.shape), buffer_ptr=feats.data_ptr())
            binding.bind_input('p_len', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(pitch_length.shape), buffer_ptr=pitch_length.data_ptr())
            binding.bind_input('sid', device_type='cuda', device_id=feats.device.index, element_type=np.int64, shape=tuple(sid.shape), buffer_ptr=sid.data_ptr())
            binding.bind_cpu_input('skip_head', np.array(skip_head, dtype=np.int64))

            binding.bind_output('audio', device_type='cuda', device_id=feats.device.index)

            self.model.run_with_iobinding(binding)

            output = [output.numpy() for output in binding.get_outputs()]
        else:
            output = self.model.run(
                ["audio"],
                {
                    "feats": feats.detach().cpu().numpy().astype(np.float16 if self.input_feats_half else np.float32, copy=False),
                    "p_len": pitch_length.detach().cpu().numpy(),
                    "sid": sid.detach().cpu().numpy(),
                    "skip_head": np.array(skip_head, dtype=np.int64)
                },
            )
        # self.model.end_profiling()

        res = torch.as_tensor(output[0], dtype=torch.float32, device=sid.device)

        if self.inferencerTypeVersion == "v2.1" or self.inferencerTypeVersion == "v2.2" or self.inferencerTypeVersion == "v1.1":
            return res
        return torch.clip(res[0, 0], -1.0, 1.0)