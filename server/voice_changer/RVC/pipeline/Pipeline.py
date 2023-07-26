import numpy as np
from typing import Any
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    DeviceChangingException,
    HalfPrecisionChangingException,
    NotEnoughDataExtimateF0,
)

from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencerNono import OnnxRVCInferencerNono

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
import logging

logger = logging.getLogger("vcclient")


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
        print("GENERATE INFERENCER", self.inferencer)
        print("GENERATE EMBEDDER", self.embedder)
        print("GENERATE PITCH EXTRACTOR", self.pitchExtractor)
        logger.info("GENERATE INFERENCER" + str(self.inferencer))
        logger.info("GENERATE EMBEDDER" + str(self.embedder))
        logger.info("GENERATE PITCH EXTRACTOR" + str(self.pitchExtractor))

        self.index = index
        self.big_npy = index.reconstruct_n(0, index.ntotal) if index is not None else None
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
        return {"inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo, "isHalf": self.isHalf}

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    def exec(
        self,
        sid,
        audio,  # torch.tensor [n]
        pitchf,  # np.array [m]
        feature,  # np.array [m, feat]
        f0_up_key,
        index_rate,
        if_f0,
        silence_front,
        embOutputLayer,
        useFinalProj,
        repeat,
        protect=0.5,
        out_size=None,
    ):
        # 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。

        search_index = self.index is not None and self.big_npy is not None and index_rate != 0
        # self.t_pad = self.sr * repeat  # 1秒
        # self.t_pad_tgt = self.targetSR * repeat  # 1秒　出力時のトリミング(モデルのサンプリングで出力される)
        audio = audio.unsqueeze(0)

        quality_padding_sec = (repeat * (audio.shape[1] - 1)) / self.sr  # padding(reflect)のサイズは元のサイズより小さい必要がある。

        self.t_pad = round(self.sr * quality_padding_sec)  # 前後に音声を追加
        self.t_pad_tgt = round(self.targetSR * quality_padding_sec)  # 前後に音声を追加　出力時のトリミング(モデルのサンプリングで出力される)
        audio_pad = F.pad(audio, (self.t_pad, self.t_pad), mode="reflect").squeeze(0)
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        # RVC QualityがOnのときにはsilence_frontをオフに。
        silence_front = silence_front if repeat == 0 else 0
        pitchf = pitchf if repeat == 0 else np.zeros(p_len)
        out_size = out_size if repeat == 0 else None

        # ピッチ検出
        try:
            if if_f0 == 1:
                pitch, pitchf = self.pitchExtractor.extract(
                    audio_pad,
                    pitchf,
                    f0_up_key,
                    self.sr,
                    self.window,
                    silence_front=silence_front,
                )
                # pitch = pitch[:p_len]
                # pitchf = pitchf[:p_len]
                pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
                pitchf = torch.tensor(pitchf, device=self.device, dtype=torch.float).unsqueeze(0)
            else:
                pitch = None
                pitchf = None
        except IndexError as e:  # NOQA
            # print(e)
            # import traceback
            # traceback.print_exc()
            raise NotEnoughDataExtimateF0()

        # tensor型調整
        feats = audio_pad
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)

        # embedding
        with autocast(enabled=self.isHalf):
            try:
                feats = self.embedder.extractFeatures(feats, embOutputLayer, useFinalProj)
                if torch.isnan(feats).all():
                    raise DeviceCannotSupportHalfPrecisionException()
            except RuntimeError as e:
                if "HALF" in e.__str__().upper():
                    raise HalfPrecisionChangingException()
                elif "same device" in e.__str__():
                    raise DeviceChangingException()
                else:
                    raise e

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

            # recover silient font
            npy = np.concatenate([np.zeros([npyOffset, npy.shape[1]], dtype=np.float32), feature[:npyOffset:2].astype("float32"), npy])[-feats.shape[1]:]
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and search_index:
            feats0 = feats.clone()

        # ピッチサイズ調整
        p_len = audio_pad.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        feats_len = feats.shape[1]
        if pitch is not None and pitchf is not None:
            pitch = pitch[:, -feats_len:]
            pitchf = pitchf[:, -feats_len:]
        p_len = torch.tensor([feats_len], device=self.device).long()

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
        if type(self.inferencer) in [OnnxRVCInferencer, OnnxRVCInferencerNono]:
            npyOffset = math.floor(silence_front * 16000) // 360
            feats = feats[:, npyOffset * 2 :, :]  # NOQA

        feats_len = feats.shape[1]
        if pitch is not None and pitchf is not None:
            pitch = pitch[:, -feats_len:]
            pitchf = pitchf[:, -feats_len:]
        p_len = torch.tensor([feats_len], device=self.device).long()

        # 推論実行
        try:
            with torch.no_grad():
                with autocast(enabled=self.isHalf):
                    audio1 = (
                        torch.clip(
                            self.inferencer.infer(feats, p_len, pitch, pitchf, sid, out_size)[0][0, 0].to(dtype=torch.float32),
                            -1.0,
                            1.0,
                        )
                        * 32767.5
                    ).data.to(dtype=torch.int16)
        except RuntimeError as e:
            if "HALF" in e.__str__().upper():
                print("11", e)
                raise HalfPrecisionChangingException()
            else:
                raise e

        feats_buffer = feats.squeeze(0).detach().cpu()
        if pitchf is not None:
            pitchf_buffer = pitchf.squeeze(0).detach().cpu()
        else:
            pitchf_buffer = None

        del p_len, pitch, pitchf, feats
        # torch.cuda.empty_cache()

        # inferで出力されるサンプリングレートはモデルのサンプリングレートになる。
        # pipelineに（入力されるときはhubertように16k）
        if self.t_pad_tgt != 0:
            offset = self.t_pad_tgt
            end = -1 * self.t_pad_tgt
            audio1 = audio1[offset:end]

        del sid
        # torch.cuda.empty_cache()
        return audio1, pitchf_buffer, feats_buffer

    def __del__(self):
        del self.embedder
        del self.inferencer
        del self.pitchExtractor
        print('Pipeline has been deleted')
