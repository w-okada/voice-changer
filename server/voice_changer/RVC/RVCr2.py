"""
VoiceChangerV2向け
"""
from dataclasses import asdict
import numpy as np
import torch
import torch.nn.functional as F
from data.ModelSlot import RVCModelSlot
from mods.log_control import VoiceChangaerLogger

from voice_changer.RVC.RVCSettings import RVCSettings
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.utils.VoiceChangerModel import (
    AudioInOutFloat,
    VoiceChangerModel,
)
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.RVC.onnxExporter.export2onnx import export2onnx
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.RVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from torchaudio import transforms as tat

from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    PipelineCreateException,
    PipelineNotInitializedException,
)
import soxr
from typing import cast

logger = VoiceChangaerLogger.get_instance().getLogger()


class RVCr2(VoiceChangerModel):
    def __init__(self, params: VoiceChangerParams, slotInfo: RVCModelSlot):
        logger.info("[Voice Changer] [RVCr2] Creating instance ")
        self.voiceChangerType = "RVC"

        self.device_manager = DeviceManager.get_instance()
        EmbedderManager.initialize(params)
        PitchExtractorManager.initialize(params)
        self.settings = RVCSettings()
        self.params = params
        # self.pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector, self.settings.gpu)

        self.pipeline: Pipeline | None = None

        self.audio_buffer: torch.Tensor | None = None
        self.pitchf_buffer: torch.Tensor | None = None
        self.feature_buffer: torch.Tensor | None = None
        self.prevVol = 0.0
        self.slotInfo = slotInfo

        self.sr = 16000
        self.window = 160

        self.resampler_in: tat.Resample | None = None
        self.resampler_out: tat.Resample | None = None

        self.input_sample_rate = 44100
        self.outputSampleRate = 44100

        self.initialize()

    def initialize(self):
        logger.info("[Voice Changer] [RVCr2] Initializing... ")

        # pipelineの生成
        try:
            self.pipeline = createPipeline(
                self.params, self.slotInfo, self.settings.gpu, self.settings.f0Detector
            )
        except PipelineCreateException as e:  # NOQA
            logger.error(
                "[Voice Changer] pipeline create failed. check your model is valid."
            )
            return

        # その他の設定
        self.settings.tran = self.slotInfo.defaultTune
        self.settings.indexRatio = self.slotInfo.defaultIndexRatio
        self.settings.protect = self.slotInfo.defaultProtect

        self.audio_buffer = torch.zeros((0,), dtype=torch.float32, device=self.device_manager.device)
        self.pitchf_buffer = torch.zeros((0,), dtype=torch.float32, device=self.device_manager.device)
        self.feature_buffer = torch.zeros((0, self.slotInfo.embChannels), dtype=torch.float32, device=self.device_manager.device)

        self.resampler_in = tat.Resample(
            orig_freq=self.input_sample_rate,
            new_freq=self.sr,
            dtype=torch.float32
        ).to(self.device_manager.device)

        self.resampler_out = tat.Resample(
            orig_freq=self.slotInfo.samplingRate,
            new_freq=self.outputSampleRate,
            dtype=torch.float32
        ).to(self.device_manager.device)

        logger.info(f"[Voice Changer] [RVCr2] Initializing on {self.device_manager.device}... done")

    def setSamplingRate(self, input_sample_rate, outputSampleRate):
        self.input_sample_rate = input_sample_rate
        self.outputSampleRate = outputSampleRate
        # self.initialize()

    def update_settings(self, key: str, val: int | float | str):
        logger.info(f"[Voice Changer] [RVCr2]: update_settings {key}:{val}")

        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "gpu":
                self.device_manager.setForceTensor(False)
                self.initialize()
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
            if key == "f0Detector" and self.pipeline is not None:
                pitchExtractor = PitchExtractorManager.getPitchExtractor(
                    self.settings.f0Detector, self.settings.gpu
                )
                self.pipeline.setPitchExtractor(pitchExtractor)
        else:
            return False
        return True

    def set_slot_info(self, slotInfo: RVCModelSlot):
        self.slotInfo = slotInfo

    def get_info(self):
        data = asdict(self.settings)
        if self.pipeline is not None:
            pipelineInfo = self.pipeline.getPipelineInfo()
            data["pipelineInfo"] = pipelineInfo
        else:
            data["pipelineInfo"] = "None"
        return data

    def get_processing_sampling_rate(self):
        return self.slotInfo.samplingRate

    def alloc(
        self,
        convert_size: int,
        convert_feature_size: int,
    ):
        # 過去のデータに連結
        if self.audio_buffer.shape[0] < convert_size:
            print('Reallocating audio buffer. Old size:', self.audio_buffer.shape[0])
            self.audio_buffer = torch.zeros(convert_size, dtype=torch.float32, device=self.device_manager.device)
        # Align audio buffer size with the latest convert size. Will be realloc'ed later if necessary,
        self.audio_buffer = self.audio_buffer[:convert_size]

        if self.slotInfo.f0:
            if self.pitchf_buffer.shape[0] < convert_feature_size:
                print('Reallocating pitchf buffer. Old size:', self.pitchf_buffer.shape[0])
                self.pitchf_buffer = torch.zeros(convert_feature_size, dtype=torch.float32, device=self.device_manager.device)
            self.pitchf_buffer = self.pitchf_buffer[:convert_feature_size]

        if self.feature_buffer.shape[0] < convert_feature_size:
            print('Reallocating feature buffer. Old size:', self.feature_buffer.shape[0])
            self.feature_buffer = torch.zeros((convert_feature_size, self.slotInfo.embChannels), dtype=torch.float32, device=self.device_manager.device)
        self.feature_buffer = self.feature_buffer[:convert_feature_size]

        return (
            self.audio_buffer,
            self.pitchf_buffer,
            self.feature_buffer,
        )

    def write_input(
        self,
        new_data: torch.Tensor,
        target: torch.Tensor,
    ):
        input_size = new_data.shape[0]
        old_data = target[input_size:].detach().clone() # Pytorch doesn't allow to move memory chunk to the same shared memory
        offset = target.shape[0] - input_size
        target[:offset] = old_data
        target[offset:] = new_data
        return target

    def inference(
        self, received_data: AudioInOutFloat, crossfade_frame: int, sola_search_frame: int
    ):
        if self.pipeline is None:
            logger.info("[Voice Changer] Pipeline is not initialized.")
            raise PipelineNotInitializedException()

        received_data = torch.as_tensor(received_data, dtype=torch.float32, device=self.device_manager.device)

        input_size = received_data.shape[0]

        vol_t = torch.sqrt(
            torch.square(received_data).mean()
        )
        vol = max(vol_t.item(), 0)
        self.prevVol = vol

        if vol < self.settings.silentThreshold:
            # FIXME: Input size is multiplied by 2 to make sure it's more than SOLA buffer
            return torch.zeros(input_size * 2, device=self.device_manager.device, dtype=torch.float32)

        received_data: torch.Tensor = self.resampler_in(received_data)
        # 処理は16Kで実施(Pitch, embed, (infer))
        # received_data: AudioInOutFloat = soxr.resample(
        #     received_data,
        #     self.input_sample_rate,
        #     self.sr,
        # )

        # received_data = torch.as_tensor(received_data, dtype=torch.float32, device=self.device_manager.device)

        crossfade_frame = int((crossfade_frame / self.input_sample_rate) * self.sr)
        sola_search_frame = int((sola_search_frame / self.input_sample_rate) * self.sr)
        extra_frame = int((self.settings.extraConvertSize / self.input_sample_rate) * self.sr)

        convert_size = input_size + crossfade_frame + sola_search_frame + extra_frame
        if (modulo := convert_size % self.window) != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convert_size = convert_size + (self.window - modulo)

        # 16k で入ってくる。
        convert_feature_size = convert_size // self.window

         # 入力データ生成
        audio, pitchf, feature = self.alloc(convert_size, convert_feature_size)

        self.audio_buffer = self.write_input(received_data, audio)

        repeat = self.settings.rvcQuality

        if repeat:
            audio = audio.unsqueeze(0)

            quality_padding_sec = (audio.shape[1] - 1) / self.sr  # padding(reflect)のサイズは元のサイズより小さい必要がある。

            t_pad = round(self.sr * quality_padding_sec)  # 前後に音声を追加
            silence_front = 0
            out_size = None

            t_pad_tgt = round(self.slotInfo.samplingRate * quality_padding_sec)  # 前後に音声を追加　出力時のトリミング(モデルのサンプリングで出力される)
            audio = F.pad(audio, (t_pad, t_pad), mode="reflect").squeeze(0)
            convert_feature_size = audio.shape[0] // self.window
        else:
            t_pad_tgt = None
            out_size = int(((convert_size - extra_frame) / self.sr) * self.slotInfo.samplingRate)
            silence_front = self.settings.extraConvertSize / self.input_sample_rate \
                if self.settings.silenceFront \
                else 0.0

        sid = self.settings.dstId
        f0_up_key = self.settings.tran
        index_rate = self.settings.indexRatio
        protect = self.settings.protect

        if_f0 = self.slotInfo.f0
        embOutputLayer = self.slotInfo.embOutputLayer
        useFinalProj = self.slotInfo.useFinalProj

        try:
            audio_out, pitchf, feature = self.pipeline.exec(
                sid,
                audio,
                pitchf,
                feature,
                f0_up_key,
                index_rate,
                if_f0,
                # 0,
                silence_front,  # extaraDataSizeの秒数。入力のサンプリングレートで算出
                embOutputLayer,
                useFinalProj,
                repeat,
                protect,
                out_size,
            )
        except DeviceCannotSupportHalfPrecisionException as e:  # NOQA
            logger.warn(
                "[Device Manager] Device cannot support half precision. Fallback to float...."
            )
            self.device_manager.setForceTensor(True)
            self.initialize()
            # raise e
            return torch.zeros(input_size * 2, device=self.device_manager.device, dtype=torch.float32)

        if pitchf is not None:
            self.pitchf_buffer = self.write_input(pitchf, self.pitchf_buffer)

        self.feature_buffer = self.write_input(feature, self.feature_buffer)

        # inferで出力されるサンプリングレートはモデルのサンプリングレートになる。
        # pipelineに（入力されるときはhubertように16k）
        if t_pad_tgt is not None:
            audio_out = audio_out[t_pad_tgt:-t_pad_tgt]

        # FIXME: Why the heck does it require another sqrt to amplify the volume?
        # result = (audio_out * torch.sqrt(vol_t)).detach().cpu().numpy()

        # result: AudioInOutFloat = soxr.resample(
        #     result,
        #     self.slotInfo.samplingRate,
        #     self.outputSampleRate
        # )

        result: torch.Tensor = self.resampler_out(audio_out * torch.sqrt(vol_t))

        return result

    def __del__(self):
        del self.pipeline

        # print("---------- REMOVING ---------------")

        # remove_path = os.path.join("RVC")
        # sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        # for key in list(sys.modules):
        #     val = sys.modules.get(key)
        #     try:
        #         file_path = val.__file__
        #         if file_path.find("RVC" + os.path.sep) >= 0:
        #             # print("remove", key, file_path)
        #             sys.modules.pop(key)
        #     except Exception:  # type:ignore
        #         # print(e)
        #         pass

    def export2onnx(self):
        modelSlot = self.slotInfo

        if modelSlot.isONNX:
            logger.warn("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": ""}

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        torch.cuda.empty_cache()
        self.initialize()

        output_file_simple = export2onnx(self.settings.gpu, modelSlot)

        return {
            "status": "ok",
            "path": f"/tmp/{output_file_simple}",
            "filename": output_file_simple,
        }

    def get_model_current(self):
        return [
            {
                "key": "defaultTune",
                "val": self.settings.tran,
            },
            {
                "key": "defaultIndexRatio",
                "val": self.settings.indexRatio,
            },
            {
                "key": "defaultProtect",
                "val": self.settings.protect,
            },
        ]
