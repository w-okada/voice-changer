import numpy as np

# import parselmouth
import torch
import torch.nn.functional as F
from Exceptions import HalfPrecisionChangingException

from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


class VC(object):
    def __init__(self, tgt_sr, device, is_half, x_pad):
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * x_pad
        self.device = device
        self.is_half = is_half

    def pipeline(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        sid,
        audio,
        f0_up_key,
        index,
        big_npy,
        index_rate,
        if_f0,
        silence_front=0,
        embChannels=256,
    ):
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        # ピッチ検出
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = pitchExtractor.extract(
                audio_pad,
                f0_up_key,
                self.sr,
                self.window,
                silence_front=silence_front,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(
                pitchf, device=self.device, dtype=torch.float
            ).unsqueeze(0)

        # tensor
        feats = torch.from_numpy(audio_pad)
        if self.is_half is True:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)

        # embedding
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        try:
            feats = embedder.extractFeatures(feats, embChannels)
        except RuntimeError as e:
            if "HALF" in e.__str__().upper():
                raise HalfPrecisionChangingException()
            else:
                raise e

        # Index - feature抽出
        if (
            isinstance(index, type(None)) is False
            and isinstance(big_npy, type(None)) is False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half is True:
                npy = npy.astype("float32")
            D, I = index.search(npy, 1)
            npy = big_npy[I.squeeze()]
            if self.is_half is True:
                npy = npy.astype("float16")

            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        #
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # ピッチ抽出
        p_len = audio_pad.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()

        # 推論実行
        with torch.no_grad():
            audio1 = (
                (inferencer.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0] * 32768)
                .data.cpu()
                .float()
                .numpy()
                .astype(np.int16)
            )

            # if pitch is not None:
            #     print("INFERENCE 1 ")
            #     audio1 = (
            #         (
            #             inferencer.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
            #             * 32768
            #         )
            #         .data.cpu()
            #         .float()
            #         .numpy()
            #         .astype(np.int16)
            #     )
            # else:
            #     if hasattr(inferencer, "infer_pitchless"):
            #         print("INFERENCE 2 ")

            #         audio1 = (
            #             (inferencer.infer_pitchless(feats, p_len, sid)[0][0, 0] * 32768)
            #             .data.cpu()
            #             .float()
            #             .numpy()
            #             .astype(np.int16)
            #         )
            #     else:
            #         print("INFERENCE 3 ")
            #         audio1 = (
            #             (inferencer.infer(feats, p_len, sid)[0][0, 0] * 32768)
            #             .data.cpu()
            #             .float()
            #             .numpy()
            #             .astype(np.int16)
            #         )

        del feats, p_len, padding_mask
        torch.cuda.empty_cache()

        if self.t_pad_tgt != 0:
            offset = self.t_pad_tgt
            end = -1 * self.t_pad_tgt
            audio1 = audio1[offset:end]

        del pitch, pitchf, sid
        torch.cuda.empty_cache()
        return audio1
