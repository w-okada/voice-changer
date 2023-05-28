import numpy as np
from typing import Any

import torch
import torch.nn.functional as F
from Exceptions import (
    DeviceChangingException,
    HalfPrecisionChangingException,
    NotEnoughDataExtimateF0,
)

from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor


# isHalfが変わる場合はPipeline作り直し
# device(GPU, isHalf変更が伴わない場合), pitchExtractorの変更は、入れ替えで対応


class Pipeline(object):
    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    index: Any | None
    big_npy: Any | None
    # feature: Any | None

    targetSR: int
    device: torch.device
    isHalf: bool

    def __init__(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        index: Any | None,
        # feature: Any | None,
        targetSR,
        device,
        isHalf,
    ):
        self.embedder = embedder
        self.inferencer = inferencer
        self.pitchExtractor = pitchExtractor

        self.index = index
        self.big_npy = (
            index.reconstruct_n(0, index.ntotal) if index is not None else None
        )
        # self.feature = feature

        self.targetSR = targetSR
        self.device = device
        self.isHalf = isHalf

        self.sr = 16000
        self.window = 160

        self.device = device
        self.isHalf = isHalf

    def setDevice(self, device: torch.device):
        self.device = device
        self.embedder.setDevice(device)
        self.inferencer.setDevice(device)

    def setDirectMLEnable(self, enable: bool):
        if hasattr(self.inferencer, "setDirectMLEnable"):
            self.inferencer.setDirectMLEnable(enable)

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    def exec(
        self,
        sid,
        audio,
        f0_up_key,
        index_rate,
        if_f0,
        silence_front,
        embOutputLayer,
        useFinalProj,
        repeat,
    ):
        self.t_pad = self.sr * repeat
        self.t_pad_tgt = self.targetSR * repeat

        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        # ピッチ検出
        pitch, pitchf = None, None
        try:
            if if_f0 == 1:
                pitch, pitchf = self.pitchExtractor.extract(
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
        except IndexError as e:
            print(e)
            raise NotEnoughDataExtimateF0()

        # tensor型調整
        feats = torch.from_numpy(audio_pad)
        if self.isHalf is True:
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
            feats = self.embedder.extractFeatures(feats, embOutputLayer, useFinalProj)
        except RuntimeError as e:
            if "HALF" in e.__str__().upper():
                raise HalfPrecisionChangingException()
            elif "same device" in e.__str__():
                raise DeviceChangingException()
            else:
                raise e

        # Index - feature抽出
        # if self.index is not None and self.feature is not None and index_rate != 0:
        if self.index is not None and self.big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.isHalf is True:
                npy = npy.astype("float32")
            # D, I = self.index.search(npy, 1)
            # npy = self.feature[I.squeeze()]

            # TODO: kは調整できるようにする
            k = 1
            if k == 1:
                _, ix = self.index.search(npy, 1)
                npy = self.big_npy[ix.squeeze()]               
            else:
                score, ix = self.index.search(npy, k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.isHalf is True:
                npy = npy.astype("float16")

            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # ピッチサイズ調整
        p_len = audio_pad.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()

        # 推論実行
        try:
            with torch.no_grad():
                audio1 = (
                    (
                        self.inferencer.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
                        * 32768
                    )
                    .data.cpu()
                    .float()
                    .numpy()
                    .astype(np.int16)
                )
        except RuntimeError as e:
            if "HALF" in e.__str__().upper():
                raise HalfPrecisionChangingException()
            else:
                raise e

        del feats, p_len, padding_mask
        torch.cuda.empty_cache()

        if self.t_pad_tgt != 0:
            offset = self.t_pad_tgt
            end = -1 * self.t_pad_tgt
            audio1 = audio1[offset:end]

        del pitch, pitchf, sid
        torch.cuda.empty_cache()
        return audio1
