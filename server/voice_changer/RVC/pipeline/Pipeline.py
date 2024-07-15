from faiss import IndexIVFFlat
import faiss.contrib.torch_utils

from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info
)

import numpy as np
import sys
import torch
import torch.nn.functional as F
import onnxruntime
from torchaudio import transforms as tat
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from mods.log_control import VoiceChangaerLogger

from voice_changer.common.TorchUtils import circular_write
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.inferencer.Inferencer import Inferencer

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.utils.Timer import Timer2
from const import F0_MEL_MIN, F0_MEL_MAX

logger = VoiceChangaerLogger.get_instance().getLogger()


class Pipeline:
    embedder: Embedder
    inferencer: Inferencer
    pitchExtractor: PitchExtractor

    index: IndexIVFFlat | None
    index_reconstruct: torch.Tensor | None
    # feature: Any | None

    model_sr: int
    device: torch.device
    isHalf: bool

    def __init__(
        self,
        embedder: Embedder,
        inferencer: Inferencer,
        pitchExtractor: PitchExtractor,
        index: IndexIVFFlat | None,
        index_reconstruct: torch.Tensor | None,
        use_f0: bool,
        model_sr: int,
        embChannels: int,
    ):
        self.embedder = embedder
        self.inferencer = inferencer
        self.pitchExtractor = pitchExtractor
        logger.info("GENERATE INFERENCER" + str(self.inferencer))
        logger.info("GENERATE EMBEDDER" + str(self.embedder))
        logger.info("GENERATE PITCH EXTRACTOR" + str(self.pitchExtractor))

        self.device_manager = DeviceManager.get_instance()
        self.device = self.device_manager.device
        self.is_half = self.device_manager.use_fp16()

        self.index = index
        self.index_reconstruct: torch.Tensor | None = index_reconstruct
        self.use_index = index is not None and self.index_reconstruct is not None
        self.use_gpu_index = sys.platform == 'linux' and '+cu' in torch.__version__ and self.device.type == 'cuda'
        self.use_f0 = use_f0

        self.onnx_upscaler = self.make_onnx_upscaler(embChannels) if self.device.type == 'privateuseone' else None

        self.model_sr = model_sr
        self.model_window = model_sr // 100

        self.dtype = torch.float16 if self.is_half else torch.float32

        self.sr = 16000
        self.window = 160

        self.resamplers = {}

    def make_onnx_upscaler(self, dim_size: int):
        # Inputs
        input = make_tensor_value_info('in', TensorProto.FLOAT16 if self.is_half else TensorProto.FLOAT, [1, dim_size, None])
        scales = make_tensor_value_info('scales', TensorProto.FLOAT, [None])
        # Outputs
        output = make_tensor_value_info('out', TensorProto.FLOAT16 if self.is_half else TensorProto.FLOAT, [1, dim_size, None])

        resize_node = make_node(
            "Resize",
            inputs=["in", "", "scales"],
            outputs=["out"],
            mode="nearest",
            axes=[2]
        )

        graph = make_graph([resize_node], 'upscaler', [input, scales], [output])

        onnx_model = make_model(graph)

        (
            providers,
            provider_options,
        ) = self.device_manager.get_onnx_execution_provider()
        return onnxruntime.InferenceSession(onnx_model.SerializeToString(), providers=providers, provider_options=provider_options)

    def getPipelineInfo(self):
        inferencerInfo = self.inferencer.getInferencerInfo() if self.inferencer else {}
        embedderInfo = self.embedder.get_embedder_info()
        pitchExtractorInfo = self.pitchExtractor.getPitchExtractorInfo()
        return {"inferencer": inferencerInfo, "embedder": embedderInfo, "pitchExtractor": pitchExtractorInfo}

    def setPitchExtractor(self, pitchExtractor: PitchExtractor):
        self.pitchExtractor = pitchExtractor

    def extractPitch(self, audio: torch.Tensor, pitch: torch.Tensor | None, pitchf: torch.Tensor | None, f0_up_key: int, formant_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = self.pitchExtractor.extract(
            audio,
            self.sr,
            self.window,
        )
        f0 *= 2 ** ((f0_up_key - formant_shift) / 12)

        f0_mel = 1127.0 * torch.log(1.0 + f0 / 700.0)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel, out=f0_mel).long()

        if pitch is not None and pitchf is not None:
            circular_write(f0_coarse, pitch)
            circular_write(f0, pitchf)
        else:
            pitch = f0_coarse
            pitchf = f0

        return pitch.unsqueeze(0), pitchf.unsqueeze(0)

    def extract_features(self, feats: torch.Tensor, embOutputLayer: int, useFinalProj: bool):
        try:
            return self.embedder.extract_features(feats, embOutputLayer, useFinalProj)
        except RuntimeError as e:
            print("Failed to extract features:", e)
            raise e

    def infer(self, feats: torch.Tensor, p_len: torch.Tensor, pitch: torch.Tensor, pitchf: torch.Tensor, sid: torch.Tensor, skip_head: int | None, return_length: int | None, formant_length: int | None) -> torch.Tensor:
        try:
            return self.inferencer.infer(feats, p_len, pitch, pitchf, sid, skip_head, return_length, formant_length)
        except RuntimeError as e:
            print("Failed to infer:", e)
            raise e

    def _search_index(self, audio: torch.Tensor, top_k: int = 1):
        if top_k == 1:
            _, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), 1)
            ix = ix.to(self.device)
            return self.index_reconstruct[ix.squeeze()]

        score, ix = self.index.search(audio if self.use_gpu_index else audio.detach().cpu(), k=top_k)
        score, ix = (
            score.to(self.device),
            ix.to(self.device),
        )
        weight = torch.square(1 / score)
        weight /= weight.sum(dim=1, keepdim=True)
        return torch.sum(self.index_reconstruct[ix] * weight.unsqueeze(2), dim=1)

    def _upscale(self, feats: torch.Tensor) -> torch.Tensor:
        if self.onnx_upscaler is not None:
            feats = self.onnx_upscaler.run(['out'], { 'in': feats.permute(0, 2, 1).detach().cpu().numpy(), 'scales': np.array([2], dtype=np.float32) })
            return torch.as_tensor(feats[0], dtype=self.dtype, device=self.device).permute(0, 2, 1).contiguous()
        return F.interpolate(feats.permute(0, 2, 1), scale_factor=2, mode='nearest').permute(0, 2, 1).contiguous()

    def exec(
        self,
        sid: int,
        audio: torch.Tensor,  # torch.tensor [n]
        pitch: torch.Tensor | None,  # torch.tensor [m]
        pitchf: torch.Tensor | None,  # torch.tensor [m]
        f0_up_key: int,
        formant_shift: float,
        index_rate: float,
        audio_feats_len: int,
        silence_front: int,
        embOutputLayer: int,
        useFinalProj: bool,
        skip_head: int,
        return_length: int,
        protect: float = 0.5,
    ) -> torch.Tensor:
        with Timer2("Pipeline-Exec", False) as t:  # NOQA
            # 16000のサンプリングレートで入ってきている。以降この世界は16000で処理。
            assert audio.dim() == 1, audio.dim()

            formant_factor = 2 ** (formant_shift / 12)
            formant_length = int(np.ceil(return_length * formant_factor))
            t.record("pre-process")

            # ピッチ検出
            pitch, pitchf = self.extractPitch(audio[silence_front:], pitch, pitchf, f0_up_key, formant_shift) if self.use_f0 else (None, None)
            t.record("extract-pitch")

            # embedding
            feats = self.extract_features(audio.view(1, -1), embOutputLayer, useFinalProj)
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            t.record("extract-feats")

            # Index - feature抽出
            is_active_index = self.use_index and index_rate > 0
            use_protect = protect < 0.5
            if self.use_f0 and is_active_index and use_protect:
                feats_orig = feats.detach().clone()

            if is_active_index:
                skip_offset = skip_head // 2
                index_audio = feats[0][skip_offset :]

                # TODO: kは調整できるようにする
                index_audio = self._search_index(index_audio.float(), 8).unsqueeze(0)
                if self.is_half:
                    index_audio = index_audio.half()

                # Recover silent front
                feats[0][skip_offset :] = index_audio * index_rate + feats[0][skip_offset :] * (1 - index_rate)

            feats = self._upscale(feats)[:, :audio_feats_len, :]
            if self.use_f0:
                pitch = pitch[:, -audio_feats_len:]
                pitchf = pitchf[:, -audio_feats_len:] * (formant_length / return_length)
                # pitchの推定が上手くいかない(pitchf=0)場合、検索前の特徴を混ぜる
                # pitchffの作り方の疑問はあるが、本家通りなので、このまま使うことにする。
                # https://github.com/w-okada/voice-changer/pull/276#issuecomment-1571336929
                if is_active_index and use_protect:
                    # FIXME: Another interpolate on feats is a big performance hit.
                    feats_orig = self._upscale(feats_orig)[:, :audio_feats_len, :]
                    pitchff = pitchf.detach().clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    pitchff = pitchff.unsqueeze(-1)
                    feats = feats * pitchff + feats_orig * (1 - pitchff)

            p_len = torch.tensor([audio_feats_len], device=self.device, dtype=torch.int64)

            sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
            t.record("mid-precess")
            # 推論実行
            out_audio = self.infer(feats, p_len, pitch, pitchf, sid, skip_head, return_length, formant_length)
            t.record("infer")

            # Formant shift sample rate adjustment
            scaled_window = int(np.floor(formant_factor * self.model_window))
            if scaled_window != self.model_window:
                if scaled_window not in self.resamplers:
                    self.resamplers[scaled_window] = tat.Resample(
                        orig_freq=scaled_window,
                        new_freq=self.model_window,
                        dtype=self.dtype,
                    ).to(self.device)
                out_audio = self.resamplers[scaled_window](
                    out_audio[: return_length * scaled_window]
                )
        # print("EXEC AVERAGE:", t.avrSecs)
        return out_audio
