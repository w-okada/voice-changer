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
from voice_changer.DiffusionSVC.inferencer.diffusion_svc_model.F0Extractor import F0_Extractor
from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor

from voice_changer.RVC.embedder.Embedder import Embedder

from voice_changer.common.VolumeExtractor import VolumeExtractor
from torchaudio.transforms import Resample


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
        resamplerIn: Resample,
        resamplerOut: Resample
    ):
        self.inferencer = inferencer
        inferencer_block_size, inferencer_sampling_rate = inferencer.getConfig()
        self.hop_size = inferencer_block_size * 16000 / inferencer_sampling_rate  # 16000はオーディオのサンプルレート。16Kで処理
        self.inferencer_block_size = inferencer_block_size
        self.inferencer_sampling_rate = inferencer_sampling_rate

        self.volumeExtractor = VolumeExtractor(self.hop_size)
        self.embedder = embedder
        self.pitchExtractor = pitchExtractor

        self.resamplerIn = resamplerIn
        self.resamplerOut = resamplerOut
        
        # self.f0ex = self.load_f0_extractor(f0_model="harvest", f0_min=50, f0_max=1100)

        print("VOLUME EXTRACTOR", self.volumeExtractor)
        print("GENERATE INFERENCER", self.inferencer)
        print("GENERATE EMBEDDER", self.embedder)
        print("GENERATE PITCH EXTRACTOR", self.pitchExtractor)

        self.targetSR = targetSR
        self.device = device
        self.isHalf = False

    def load_f0_extractor(self, f0_model, f0_min=None, f0_max=None):
        f0_extractor = F0_Extractor(
            f0_extractor=f0_model,
            sample_rate=44100,
            hop_size=512,
            f0_min=f0_min,
            f0_max=f0_max,
            block_size=512,
            model_sampling_rate=44100
        )
        return f0_extractor

    def getPipelineInfo(self):
        volumeExtractorInfo = self.volumeExtractor.getVolumeExtractorInfo()
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.getEmbedderInfo()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {"volumeExtractor": volumeExtractorInfo, "inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo, "isHalf": self.isHalf}

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    @torch.no_grad()
    def extract_volume_and_mask(self, audio: torch.Tensor, threshold: float):
        '''
        with Timer("[VolumeExt np]") as t:
            for i in range(100):
                volume = self.volumeExtractor.extract(audio)
        time_np = t.secs
        with Timer("[VolumeExt pt]") as t:
            for i in range(100):
                volume_t = self.volumeExtractor.extract_t(audio)
        time_pt = t.secs

        print("[Volume np]:", volume)
        print("[Volume pt]:", volume_t)
        print("[Perform]:", time_np, time_pt)
        # -> [Perform]: 0.030178070068359375 0.005780220031738281 (RTX4090)
        # -> [Perform]: 0.029046058654785156 0.0025115013122558594 (CPU i9 13900KF)
        # ---> これくらいの処理ならCPU上のTorchでやった方が早い？
        '''
        volume_t = self.volumeExtractor.extract_t(audio)
        mask = self.volumeExtractor.get_mask_from_volume_t(volume_t, self.inferencer_block_size, threshold=threshold)
        volume = volume_t.unsqueeze(-1).unsqueeze(0)
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
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        audio16k = self.resamplerIn(audio_t)
        volume, mask = self.extract_volume_and_mask(audio16k, threshold=-60.0)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        n_frames = int(audio16k.size(-1) // self.hop_size + 1)

        # ピッチ検出
        try:
            pitch = self.pitchExtractor.extract(
                audio16k.squeeze(),
                pitchf,
                f0_up_key,
                16000,                 # 音声のサンプリングレート(既に16000)
                int(self.hop_size),    # 処理のwindowサイズ (44100における512)
                silence_front=silence_front,
            )

            pitch = torch.tensor(pitch[-n_frames:], device=self.device).unsqueeze(0).long()
        except IndexError as e:  # NOQA
            raise NotEnoughDataExtimateF0()

        # tensor型調整
        feats = audio16k.squeeze()
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

        # 推論実行
        try:
            with torch.no_grad():
                with autocast(enabled=self.isHalf):
                    print("[EMBEDDER EXTRACT:::]", feats.shape, pitch.unsqueeze(-1).shape, volume.shape, mask.shape)
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
        if pitch is not None:
            pitch_buffer = pitch.squeeze(0).detach().cpu()
        else:
            pitch_buffer = None

        del pitch, pitchf, feats, sid
        torch.cuda.empty_cache()
        audio1 = self.resamplerOut(audio1.float())
        return audio1, pitch_buffer, feats_buffer
