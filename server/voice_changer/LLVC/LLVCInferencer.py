import numpy as np
import torch
import json
from voice_changer.LLVC.model.llvc import Net

from voice_changer.utils.VoiceChangerModel import AudioInOutFloat


class LLVCInferencer:
    def loadModel(self, checkpoint_path: str, config_path: str):
        with open(config_path) as f:
            config = json.load(f)
        model = Net(**config["model_params"])
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])

        self.config = config
        self.model = model

        self.enc_buf, self.dec_buf, self.out_buf = self.model.init_buffers(1, torch.device("cpu"))

        if hasattr(self.model, "convnet_pre"):
            self.convnet_pre_ctx = self.model.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        else:
            self.convnet_pre_ctx = None

        self.audio_buffer: AudioInOutFloat = np.zeros(0, dtype=np.float32)
        self.front_ctx: AudioInOutFloat | None = None

        return self

    def infer(
        self,
        audio: AudioInOutFloat,
    ) -> torch.Tensor:
        # print(f"[infer] inputsize:{audio.shape} + rest:{self.audio_buffer.shape}")
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        # print(f"[infer] concat size", self.audio_buffer.shape)

        try:
            L = self.model.L
            processing_unit = self.model.dec_chunk_size * L
            chunk_size = (len(self.audio_buffer) // processing_unit) * processing_unit

            chunk = self.audio_buffer[:chunk_size]
            self.audio_buffer = self.audio_buffer[chunk_size:]

            inputTensor = torch.from_numpy(chunk.astype(np.float32)).to("cpu")

            if self.front_ctx is None:
                inputTensor = torch.cat([torch.zeros(L * 2), inputTensor])
            else:
                inputTensor = torch.cat([self.front_ctx, inputTensor])
            self.front_ctx = inputTensor[-L * 2 :]

            audio1, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = self.model(
                inputTensor.unsqueeze(0).unsqueeze(0),
                self.enc_buf,
                self.dec_buf,
                self.out_buf,
                self.convnet_pre_ctx,
                pad=(not self.model.lookahead),
            )
            # print(f"[infer] input chunk size {chunk.shape} ->(+32) lookaheadsize{inputTensor.shape}->(same chunk) inferedsize{audio1.shape}")

            audio1 = audio1.squeeze(0).squeeze(0)
            return audio1
        except Exception as e:
            raise RuntimeError(f"Exeption in {self.__class__.__name__}", e)

    # def isTorch(self):
    #     return True
