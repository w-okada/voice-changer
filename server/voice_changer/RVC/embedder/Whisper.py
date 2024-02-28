import torch
from torch import device
from voice_changer.RVC.embedder.Embedder import Embedder

from voice_changer.RVC.embedder.whisper.audio import log_mel_spectrogram
from .whisper.whisper import load_model
import numpy as np
import torch.nn.functional as F


class Whisper(Embedder):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().setProps("whisper", file, dev, isHalf)

        whisper = load_model(file).to(dev)

        self.model = whisper
        return self

    def extractFeatures(self, audio: torch.Tensor) -> torch.Tensor:
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio.astype(np.float32))
            audio = audio.to(self.dev)
            # if self.isHalf and audio.dtype != torch.float16:
            #     audio = audio.half()
            if self.isHalf is False and audio.dtype != torch.float32:
                audio = audio.float()

            if audio.dim() != 1:
                audio = audio.squeeze(0)

            if audio.dim() != 1:
                raise RuntimeError(f"Exeption in {self.__class__.__name__} audio.dim is not 1 (size :{audio.dim()}, {audio.shape})")

            audln = audio.shape[0]
            ppgln = audln // 320

            mel = log_mel_spectrogram(audio).to(self.model.device)

            # print(f"[whisper_ppg] audio:{audio.shape}({audio.shape[0]/16000}ms) -> ppg:{ppgln}")
            # print(f"[whisper_ppg] mel:{mel.shape}({mel.dtype})")
            with torch.no_grad():
                ppg = self.model.encoder(mel.unsqueeze(0))
                padding = (0, 384)
                ppg_padded = F.pad(ppg, padding, "constant", 0)
                ppg_padded = ppg_padded.data
                # print(f"[whisper_ppg] ppg:{ppg.shape}")
        except Exception as e:
            print(e)
            raise RuntimeError(f"Exeption in {self.__class__.__name__}", e)
            # raise EmbedderProcessException(f"Exeption in {self.__class__.__name__}", e)
        return ppg_padded
