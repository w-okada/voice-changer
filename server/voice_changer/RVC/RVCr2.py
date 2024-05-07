"""
VoiceChangerV2向け
"""
from dataclasses import asdict
import torch
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
        self.prev_vol = 0.0
        self.return_length = 0
        self.skip_head = 0
        self.silence_front = 0.0
        self.slotInfo = slotInfo

        # 処理は16Kで実施(Pitch, embed, (infer))
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

        # 処理は16Kで実施(Pitch, embed, (infer))
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
        if self.input_sample_rate != input_sample_rate:
            self.input_sample_rate = input_sample_rate
            self.resampler_in = tat.Resample(
                orig_freq=self.input_sample_rate,
                new_freq=self.sr,
                dtype=torch.float32
            ).to(self.device_manager.device)
        if self.outputSampleRate != outputSampleRate:
            self.outputSampleRate = outputSampleRate
            self.resampler_out = tat.Resample(
                orig_freq=self.slotInfo.samplingRate,
                new_freq=self.outputSampleRate,
                dtype=torch.float32
            ).to(self.device_manager.device)

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

    def realloc(self, block_frame: int, extra_frame: int, crossfade_frame: int, sola_buffer_frame: int, sola_search_frame: int):
        # Calculate frame sizes based on DEVICE sample rate (f.e., 48000Hz)
        block_frame_sec = block_frame / self.input_sample_rate
        crossfade_frame_sec = crossfade_frame / self.input_sample_rate
        sola_buffer_frame_sec = sola_buffer_frame / self.input_sample_rate
        sola_search_frame_sec = sola_search_frame / self.input_sample_rate
        extra_frame_sec = extra_frame / self.input_sample_rate

        # Calculate frame sizes for 16000Hz
        block_frame_16k = int(block_frame_sec * self.sr)
        crossfade_frame_16k = int(crossfade_frame_sec * self.sr)
        sola_search_frame_16k = int(sola_search_frame_sec * self.sr)
        extra_frame_16k = int(extra_frame_sec * self.sr)

        convert_size_16k = block_frame_16k + sola_search_frame_16k + extra_frame_16k + crossfade_frame_16k
        if (modulo := convert_size_16k % self.window) != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convert_size_16k = convert_size_16k + (self.window - modulo)
        convert_feature_size_16k = convert_size_16k // self.window

        # Calculate frame sizes for MODEL INFERENCE sample rate (f.e., 32000Hz)
        model_window = self.slotInfo.samplingRate // 100
        block_frame_model = int(block_frame_sec * self.slotInfo.samplingRate)
        sola_buffer_frame_model = int(sola_buffer_frame_sec * self.slotInfo.samplingRate)
        sola_search_frame_model = int(sola_search_frame_sec * self.slotInfo.samplingRate)
        extra_frame_model = int(extra_frame_sec * self.slotInfo.samplingRate)

        # Calculate offsets for inferencer
        self.skip_head = extra_frame_model // model_window
        # FIXME: Not sure if it's still necessary to round up the return length
        self.return_length = block_frame_model + sola_buffer_frame_model + sola_search_frame_model
        if (modulo := self.return_length % model_window) != 0:
            self.return_length += model_window - modulo
        self.return_length //= model_window

        self.silence_front = extra_frame_16k if self.settings.silenceFront else 0

        self.crop_start = -(block_frame_16k + crossfade_frame_16k)
        self.crop_end = -crossfade_frame_16k

        self.audio_buffer = torch.zeros(convert_size_16k, dtype=torch.float32, device=self.device_manager.device)
        self.pitchf_buffer = torch.zeros(convert_feature_size_16k, dtype=torch.float32, device=self.device_manager.device)
        print('Allocated audio buffer:', self.audio_buffer.shape[0])
        print('Allocated pitchf buffer:', self.pitchf_buffer.shape[0])

    def write_input(
        self,
        new_data: torch.Tensor,
        target: torch.Tensor,
    ):
        offset = new_data.shape[0]
        target[: -offset] = target[offset :].detach().clone()
        target[-offset :] = new_data
        return target

    def inference(self, audio_in: AudioInOutFloat):
        if self.pipeline is None:
            raise PipelineNotInitializedException()

        audio_in_16k = self.resampler_in(
            torch.as_tensor(audio_in, dtype=torch.float32, device=self.device_manager.device)
        )

        self.audio_buffer = self.write_input(audio_in_16k, self.audio_buffer)

        vol_t = torch.sqrt(
            torch.square(self.audio_buffer[self.crop_start:self.crop_end]).mean()
        )
        vol = max(vol_t.item(), 0)
        self.prev_vol = vol

        if vol < self.settings.silentThreshold:
            return None

        try:
            audio_model, pitchf = self.pipeline.exec(
                self.settings.dstId,
                self.audio_buffer,
                self.pitchf_buffer,
                self.settings.tran,
                self.settings.indexRatio,
                self.slotInfo.f0,
                self.silence_front,
                self.slotInfo.embOutputLayer,
                self.slotInfo.useFinalProj,
                self.settings.protect,
                self.skip_head,
                self.return_length,
            )
        except DeviceCannotSupportHalfPrecisionException as e:  # NOQA
            logger.warn(
                "[Device Manager] Device cannot support half precision. Fallback to float...."
            )
            self.device_manager.setForceTensor(True)
            self.initialize()
            # raise e
            return None

        if pitchf is not None:
            self.pitchf_buffer = self.write_input(pitchf, self.pitchf_buffer)

        # FIXME: Why the heck does it require another sqrt to amplify the volume?
        audio_out: torch.Tensor = self.resampler_out(audio_model * torch.sqrt(vol_t))

        return audio_out

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
