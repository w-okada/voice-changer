import numpy as np
from typing import Any
import math
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

    def getPipelineInfo(self):
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.getEmbedderInfo()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {
            "inferencer": inferencerInfo,
            "embedder": embedderInfo,
            "pitchExtractor": pitchExtractorInfo,
        }

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
        protect=0.5,
    ):
        # 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。

        search_index = (
            self.index is not None and self.big_npy is not None and index_rate != 0
        )
        self.t_pad = self.sr * repeat
        self.t_pad_tgt = self.targetSR * repeat
        audio_pad = F.pad(
            audio.unsqueeze(0), (self.t_pad, self.t_pad), mode="reflect"
        ).squeeze(0)
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
        except IndexError:
            # print(e)
            raise NotEnoughDataExtimateF0()

        # tensor型調整
        feats = audio_pad
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
        if protect < 0.5 and search_index:
            feats0 = feats.clone()

        # Index - feature抽出
        # if self.index is not None and self.feature is not None and index_rate != 0:
        if search_index:
            npy = feats[0].cpu().numpy()
            # apply silent front for indexsearch
            npyOffset = math.floor(silence_front * 16000) // 360
            npy = npy[npyOffset:]

            if self.isHalf is True:
                npy = npy.astype("float32")

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

            # recover silient font
            npy = np.concatenate([np.zeros([npyOffset, npy.shape[1]]), npy])

            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and search_index:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )

        # ピッチサイズ調整
        p_len = audio_pad.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        # pitchの推定が上手くいかない(pitchf=0)場合、検索前の特徴を混ぜる
        # pitchffの作り方の疑問はあるが、本家通りなので、このまま使うことにする。
        # https://github.com/w-okada/voice-changer/pull/276#issuecomment-1571336929
        if protect < 0.5 and search_index:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()

        # apply silent front for inference
        npyOffset = math.floor(silence_front * 16000) // 360
        feats = feats[:, npyOffset * 2 :, :]
        feats_len = feats.shape[1]
        pitch = pitch[:, -feats_len:]
        pitchf = pitchf[:, -feats_len:]
        p_len = torch.tensor([feats_len], device=self.device).long()

        # 推論実行
        try:
            with torch.no_grad():
                audio1 = (
                    torch.clip(
                        self.inferencer.infer(feats, p_len, pitch, pitchf, sid)[0][
                            0, 0
                        ].to(dtype=torch.float32),
                        -1.0,
                        1.0,
                    )
                    * 32767.5
                    - 0.5
                ).data.to(dtype=torch.int16)
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
