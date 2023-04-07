import onnxruntime
import torch
import numpy as np
# providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
providers = ["CPUExecutionProvider"]


class ModelWrapper:
    def __init__(self, onnx_model, is_half):
        self.onnx_model = onnx_model
        self.is_half = is_half

        # ort_options = onnxruntime.SessionOptions()
        # ort_options.intra_op_num_threads = 8
        self.onnx_session = onnxruntime.InferenceSession(
            self.onnx_model,
            providers=providers
        )
        # input_info = s

    def set_providers(self, providers, provider_options=[{}]):
        self.onnx_session.set_providers(providers=providers, provider_options=provider_options)

    def get_providers(self):
        return self.onnx_session.get_providers()

    def infer(self, feats, p_len, pitch, pitchf, sid):
        if self.is_half:
            print("feats", feats.cpu().numpy().dtype)
            print("p_len", p_len.cpu().numpy().dtype)
            print("pitch", pitch.cpu().numpy().dtype)
            print("pitchf", pitchf.cpu().numpy().dtype)
            print("sid", sid.cpu().numpy().dtype)

            audio1 = self.onnx_session.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float16),
                    "p_len": p_len.cpu().numpy().astype(np.int64),
                    "pitch": pitch.cpu().numpy().astype(np.int64),
                    "pitchf": pitchf.cpu().numpy().astype(np.float32),
                    "sid": sid.cpu().numpy().astype(np.int64),
                })
        else:
            audio1 = self.onnx_session.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float32),
                    "p_len": p_len.cpu().numpy(),
                    "pitch": pitch.cpu().numpy(),
                    "pitchf": pitchf.cpu().numpy(),
                    "sid": sid.cpu().numpy(),
                })

        return torch.tensor(np.array(audio1))
