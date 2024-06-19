import torch
import numpy as np
from const import EnumInferenceTypes

from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer


class OnnxRVCInferencerNono(OnnxRVCInferencer):
    def load_model(self, file: str, inferencerTypeVersion: str | None = None):
        super().load_model(file, inferencerTypeVersion)
        self.set_props(EnumInferenceTypes.onnxRVCNono, file)
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        skip_head: int | None,
    ) -> torch.Tensor:
        if feats.device.type == 'cuda':
            binding = self.model.io_binding()

            binding.bind_input('feats', device_type='cuda', device_id=feats.device.index, element_type=self.fp_dtype_np, shape=tuple(feats.shape), buffer_ptr=feats.float().data_ptr())
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
                    "feats": feats.float().detach().cpu().numpy(),
                    "p_len": pitch_length.detach().cpu().numpy(),
                    "sid": sid.detach().cpu().numpy(),
                    "skip_head": np.array(skip_head, dtype=np.int64)
                },
            )
        # self.model.end_profiling()

        res = torch.as_tensor(output[0], dtype=self.fp_dtype_t, device=feats.device)
        if self.isHalf:
            res = res.float()

        if self.inferencerTypeVersion == "v2.1" or self.inferencerTypeVersion == "v2.2" or self.inferencerTypeVersion == "v1.1":
            return res
        return torch.clip(res[0, 0], -1.0, 1.0)