from typing import Any
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    DeviceChangingException,
    HalfPrecisionChangingException,
    NotEnoughDataExtimateF0,
)
from voice_changer.DiffusionSVC.inferencer.Inferencer import Inferencer

from voice_changer.RVC.embedder.Embedder import Embedder

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.common.VolumeExtractor import VolumeExtractor


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
        # index: Any | None,
        targetSR,
        device,
        isHalf,
    ):
        self.inferencer = inferencer
        inferencer_block_size, inferencer_sampling_rate = inferencer.getConfig()
        self.hop_size = inferencer_block_size * 16000 / inferencer_sampling_rate  # 16000はオーディオのサンプルレート。この時点で16Kになっている。
        self.inferencer_block_size = inferencer_block_size
        self.inferencer_sampling_rate = inferencer_sampling_rate

        self.volumeExtractor = VolumeExtractor(self.hop_size)
        self.embedder = embedder
        self.pitchExtractor = pitchExtractor

        print("VOLUME EXTRACTOR", self.volumeExtractor)
        print("GENERATE INFERENCER", self.inferencer)
        print("GENERATE EMBEDDER", self.embedder)
        print("GENERATE PITCH EXTRACTOR", self.pitchExtractor)

        self.targetSR = targetSR
        self.device = device
        self.isHalf = False

    def getPipelineInfo(self):
        volumeExtractorInfo = self.volumeExtractor.getVolumeExtractorInfo()
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.getEmbedderInfo()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {"volumeExtractor": volumeExtractorInfo, "inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo, "isHalf": self.isHalf}

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    @torch.no_grad()
    def extract_volume_and_mask(self, audio, threhold):
        volume = self.volumeExtractor.extract(audio)
        mask = self.volumeExtractor.get_mask_from_volume(volume, self.inferencer_block_size, threhold=threhold, device=self.device)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        return volume, mask

    def exec(
        self,
        sid,
        audio,  # torch.tensor [n]
        pitchf,  # np.array [m]
        feature,  # np.array [m, feat]
        f0_up_key,
        silence_front,
        embOutputLayer,
        useFinalProj,
        protect=0.5
    ):
        # 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。
        audio = audio.unsqueeze(0)
        self.t_pad = 0
        audio_pad = F.pad(audio, (self.t_pad, self.t_pad), mode="reflect").squeeze(0)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        n_frames = int(audio_pad.size(-1) // self.hop_size + 1)
        volume, mask = self.extract_volume_and_mask(audio, threhold=-60.0)
        # ピッチ検出
        try:
            pitch, pitchf = self.pitchExtractor.extract(
                audio_pad,
                pitchf,
                f0_up_key,
                16000,                 # 音声のサンプリングレート(既に16000)
                # int(self.hop_size),    # 処理のwindowサイズ (44100における512)
                int(self.hop_size),    # 処理のwindowサイズ (44100における512)
                silence_front=silence_front,
            )
            print("[Pitch]", pitch)

            pitch = torch.tensor(pitch[-n_frames:], device=self.device).unsqueeze(0).long()  # 160window sizeを前提にバッファを作っているので切る。
            pitchf = torch.tensor(pitchf[-n_frames:], device=self.device, dtype=torch.float).unsqueeze(0)  # 160window sizeを前提にバッファを作っているので切る。
        except IndexError as e:  # NOQA
            # print(e)
            raise NotEnoughDataExtimateF0()

        # tensor型調整
        feats = audio_pad
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
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
        feats = F.interpolate(feats.permute(0, 2, 1), size=int(n_frames), mode='nearest').permute(0, 2, 1)

        if protect < 0.5:
            feats0 = feats.clone()

        # # ピッチサイズ調整
        # p_len = audio_pad.shape[0] // self.window
        # feats_len = feats.shape[1]
        # if feats.shape[1] < p_len:
        #     p_len = feats_len
        #     pitch = pitch[:, :feats_len]
        #     pitchf = pitchf[:, :feats_len]

        # pitch = pitch[:, -feats_len:]
        # pitchf = pitchf[:, -feats_len:]
        # p_len = torch.tensor([feats_len], device=self.device).long()

        # print("----------plen::1:", p_len)

        # pitchの推定が上手くいかない(pitchf=0)場合、検索前の特徴を混ぜる
        # pitchffの作り方の疑問はあるが、本家通りなので、このまま使うことにする。
        # https://github.com/w-okada/voice-changer/pull/276#issuecomment-1571336929
        if protect < 0.5:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        # # apply silent front for inference
        # if type(self.inferencer) in [OnnxRVCInferencer, OnnxRVCInferencerNono]:
        #     npyOffset = math.floor(silence_front * 16000) // 360  # 160x2 = 360
        #     feats = feats[:, npyOffset * 2 :, :]  # NOQA

        # 推論実行
        try:
            with torch.no_grad():
                with autocast(enabled=self.isHalf):
                    audio1 = (
                        torch.clip(
                            self.inferencer.infer(
                                feats,
                                pitch.unsqueeze(-1),
                                volume,
                                mask,
                                sid,
                                infer_speedup=10,
                                k_step=20,
                                silence_front=silence_front
                                ).to(dtype=torch.float32),
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

        del pitch, pitchf, feats, sid
        torch.cuda.empty_cache()

        return audio1, pitchf_buffer, feats_buffer
