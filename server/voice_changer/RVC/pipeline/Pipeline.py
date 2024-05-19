from faiss import Index
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    DeviceChangingException,
    HalfPrecisionChangingException,
    NotEnoughDataExtimateF0,
)
from mods.log_control import VoiceChangaerLogger

from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencer import OnnxRVCInferencer
from voice_changer.RVC.inferencer.OnnxRVCInferencerNono import OnnxRVCInferencerNono

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.utils.Timer import Timer2

logger = VoiceChangaerLogger.get_instance().getLogger()


class Pipeline:
    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    index: Index | None
    index_reconstruct: torch.Tensor | None
    # feature: Any | None

    targetSR: int
    device: torch.device
    isHalf: bool

    def __init__(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        index: Index | None,
        # feature: Any | None,
        targetSR: int,
        device: torch.device,
        isHalf: bool,
    ):
        self.embedder = embedder
        self.inferencer = inferencer
        self.pitchExtractor = pitchExtractor
        logger.info("GENERATE INFERENCER" + str(self.inferencer))
        logger.info("GENERATE EMBEDDER" + str(self.embedder))
        logger.info("GENERATE PITCH EXTRACTOR" + str(self.pitchExtractor))

        self.index = index
        self.index_reconstruct: torch.Tensor | None = torch.as_tensor(index.reconstruct_n(0, index.ntotal), dtype=torch.float32, device=device) if index is not None else None
        self.use_gpu_index = sys.platform == 'linux' and device.type == 'cuda'
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

    def extractPitch(self, audio: torch.Tensor, pitchf: torch.Tensor, f0_up_key: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        try:
            return self.pitchExtractor.extract(
                audio,
                pitchf,
                f0_up_key,
                self.sr,
                self.window,
            )
        except IndexError as e:  # NOQA
            print(e)
            import traceback
            traceback.print_exc()
            raise NotEnoughDataExtimateF0()

    def extractFeatures(self, feats: torch.Tensor, embOutputLayer: int, useFinalProj: bool):
        try:
            feats = self.embedder.extractFeatures(feats, embOutputLayer, useFinalProj)
            if torch.isnan(feats).all():
                raise DeviceCannotSupportHalfPrecisionException()
            return feats
        except RuntimeError as e:
            print("Failed to extract features:", e)
            if "HALF" in e.__str__().upper():
                raise HalfPrecisionChangingException()
            elif "same device" in e.__str__():
                raise DeviceChangingException()
            else:
                raise e

    def infer(self, feats: torch.Tensor, p_len: torch.Tensor, pitch: torch.Tensor, pitchf: torch.Tensor, sid: torch.Tensor, skip_head: int | None, return_length: int | None) -> torch.Tensor:
        try:
            return self.inferencer.infer(feats, p_len, pitch, pitchf, sid, skip_head, return_length)
        except RuntimeError as e:
            print("Failed to infer:", e)
            if "HALF" in e.__str__().upper():
                raise HalfPrecisionChangingException()
            else:
                raise e

    def _search_index(self, audio: torch.Tensor, k: int = 1):
        if k == 1:
            _, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), 1)
            ix = torch.as_tensor(ix, dtype=torch.int64, device=self.device)
            return self.index_reconstruct[ix.squeeze()]

        score, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), k=8)
        score, ix = (
            torch.as_tensor(score, dtype=torch.float32, device=self.device),
            torch.as_tensor(ix, dtype=torch.int64, device=self.device)
        )
        weight = torch.square(1 / score)
        weight /= weight.sum(dim=1, keepdim=True)
        return torch.sum(self.index_reconstruct[ix] * weight.unsqueeze(2), dim=1)


    def exec(
        self,
        sid: int,
        audio: torch.Tensor,  # torch.tensor [n]
        pitchf: torch.Tensor,  # torch.tensor [m]
        f0_up_key: int,
        index_rate: float,
        if_f0: bool,
        silence_front: int,
        embOutputLayer: int,
        useFinalProj: bool,
        protect: float = 0.5,
        skip_head: int | None = None,
        return_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        with Timer2("Pipeline-Exec", False) as t:  # NOQA
            # 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。

            # tensor型調整
            # if audio.dim() == 2:  # double channels
            #     audio = audio.mean(-1)
            assert audio.dim() == 1, audio.dim()
            feats = audio.view(1, -1)
            t.record("pre-process")

            # ピッチ検出
            # with autocast(enabled=self.isHalf):
            pitch, pitchf = self.extractPitch(audio[silence_front:], pitchf, f0_up_key) if if_f0 else (None, None)
            t.record("extract-pitch")

            # embedding
            # with autocast(enabled=self.isHalf):
            feats = self.extractFeatures(feats, embOutputLayer, useFinalProj)
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            t.record("extract-feats")

            # Index - feature抽出
            if self.index is not None and self.index_reconstruct is not None and index_rate != 0:
                skip_offset = skip_head // 2
                index_audio = feats[0][skip_offset :]

                if self.isHalf:
                    index_audio = index_audio.to(dtype=torch.float32, copy=False)

                # TODO: kは調整できるようにする
                index_audio = self._search_index(index_audio, 8)

                # Recover silent front
                feats[0][skip_offset :] = index_audio.unsqueeze(0) * index_rate + (1 - index_rate) * feats[0][skip_offset :]

                # pitchの推定が上手くいかない(pitchf=0)場合、検索前の特徴を混ぜる
                # pitchffの作り方の疑問はあるが、本家通りなので、このまま使うことにする。
                # https://github.com/w-okada/voice-changer/pull/276#issuecomment-1571336929
                if protect < 0.5:
                    pitchff = pitchf.detach().clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    pitchff = pitchff.unsqueeze(-1)
                    feats = feats * pitchff + feats * (1 - pitchff)

            audio_feats_len = audio.shape[0] // self.window
            feats: torch.Tensor = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1).contiguous()

            feats = feats[:, :audio_feats_len, :]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, -audio_feats_len:]
                pitchf = pitchf[:, -audio_feats_len:]
            p_len = torch.as_tensor([audio_feats_len], device=self.device, dtype=torch.int64)

            sid = torch.as_tensor(sid, device=self.device, dtype=torch.int64).unsqueeze(0)
            t.record("mid-precess")
            # 推論実行
            out_audio = self.infer(feats, p_len, pitch, pitchf, sid, skip_head, return_length)
            t.record("infer")

            pitchf_buffer = pitchf.squeeze(0) if pitchf is not None else None

            # del p_len, pitch, pitchf, feats, sid
            # torch.cuda.empty_cache()

            t.record("post-process")
            # torch.cuda.empty_cache()
        # print("EXEC AVERAGE:", t.avrSecs)
        return out_audio, pitchf_buffer
