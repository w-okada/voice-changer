"""
VoiceChangerV2向け
"""
from dataclasses import asdict
import torch
from data.ModelSlot import RVCModelSlot
from mods.log_control import VoiceChangaerLogger

from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.utils.VoiceChangerModel import (
    AudioInOutFloat,
    VoiceChangerModel,
)
from settings import ServerSettings
from voice_changer.RVC.onnxExporter.export2onnx import export2onnx
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.RVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.common.TorchUtils import circular_write
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from torchaudio import transforms as tat
from voice_changer.VoiceChangerSettings import VoiceChangerSettings
from Exceptions import (
    PipelineCreateException,
    PipelineNotInitializedException,
)

logger = VoiceChangaerLogger.get_instance().getLogger()


class RVCr2(VoiceChangerModel):
    def __init__(self, params: ServerSettings, slotInfo: RVCModelSlot, settings: VoiceChangerSettings):
        logger.info("[Voice Changer] [RVCr2] Creating instance")
        self.voiceChangerType = "RVC"

        self.device_manager = DeviceManager.get_instance()
        EmbedderManager.initialize(params)
        PitchExtractorManager.initialize(params)
        self.settings = settings
        self.params = params

        self.pipeline: Pipeline | None = None

        self.convert_buffer: torch.Tensor | None = None
        self.pitch_buffer: torch.Tensor | None = None
        self.pitchf_buffer: torch.Tensor | None = None
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

        self.is_half = False

        self.initialize()

    def initialize(self, force_reload: bool = False):
        logger.info("[Voice Changer] [RVCr2] Initializing... ")

        self.is_half = self.device_manager.use_fp16()

        # pipelineの生成
        try:
            self.pipeline = createPipeline(
                self.params, self.slotInfo, self.settings.f0Detector, force_reload
            )
        except PipelineCreateException as e:  # NOQA
            logger.error(
                "[Voice Changer] pipeline create failed. check your model is valid."
            )
            return

        self.dtype = torch.float16 if self.is_half else torch.float32

        # Settings update works and is reflected correctly
        # because RVCr2.initialize() is called during modelSlotIndex update.
        self.settings.set_properties({
            'tran': self.slotInfo.defaultTune,
            'formantShift': self.slotInfo.defaultFormantShift,
            'indexRatio': self.slotInfo.defaultIndexRatio,
            'protect': self.slotInfo.defaultProtect
        })

        # 処理は16Kで実施(Pitch, embed, (infer))
        self.resampler_in = tat.Resample(
            orig_freq=self.input_sample_rate,
            new_freq=self.sr,
            dtype=self.dtype
        ).to(self.device_manager.device)

        self.resampler_out = tat.Resample(
            orig_freq=self.slotInfo.samplingRate,
            new_freq=self.outputSampleRate,
            dtype=torch.float32
        ).to(self.device_manager.device)

    def setSamplingRate(self, input_sample_rate, outputSampleRate):
        if self.input_sample_rate != input_sample_rate:
            self.input_sample_rate = input_sample_rate
            self.resampler_in = tat.Resample(
                orig_freq=self.input_sample_rate,
                new_freq=self.sr,
                dtype=self.dtype
            ).to(self.device_manager.device)
        if self.outputSampleRate != outputSampleRate:
            self.outputSampleRate = outputSampleRate
            self.resampler_out = tat.Resample(
                orig_freq=self.slotInfo.samplingRate,
                new_freq=self.outputSampleRate,
                dtype=torch.float32
            ).to(self.device_manager.device)

    def update_settings(self, key: str, val, old_val):
        logger.info(f"[Voice Changer] [RVCr2]: update_settings {key}:{val}")

        if key in {"gpu", "forceFp32"}:
            self.initialize(True)
        elif key == "f0Detector" and self.pipeline is not None:
            pitchExtractor = PitchExtractorManager.getPitchExtractor(
                self.settings.f0Detector, self.settings.gpu
            )
            self.pipeline.setPitchExtractor(pitchExtractor)

    def set_slot_info(self, slotInfo: RVCModelSlot):
        self.slotInfo = slotInfo

    def get_info(self):
        data = {}
        if self.pipeline is not None:
            pipelineInfo = self.pipeline.getPipelineInfo()
            data["pipelineInfo"] = pipelineInfo
        else:
            data["pipelineInfo"] = "None"
        return data

    def get_processing_sampling_rate(self):
        return self.slotInfo.samplingRate

    def realloc(self, block_frame: int, extra_frame: int, crossfade_frame: int, sola_search_frame: int):
        # Calculate frame sizes based on DEVICE sample rate (f.e., 48000Hz) and convert to 16000Hz
        block_frame_16k = int(block_frame / self.input_sample_rate * self.sr)
        crossfade_frame_16k = int(crossfade_frame / self.input_sample_rate * self.sr)
        sola_search_frame_16k = int(sola_search_frame / self.input_sample_rate * self.sr)
        extra_frame_16k = int(extra_frame / self.input_sample_rate * self.sr)

        convert_size_16k = block_frame_16k + sola_search_frame_16k + extra_frame_16k + crossfade_frame_16k
        if (modulo := convert_size_16k % self.window) != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convert_size_16k = convert_size_16k + (self.window - modulo)
        self.convert_feature_size_16k = convert_size_16k // self.window

        self.skip_head = extra_frame_16k // self.window
        self.return_length = self.convert_feature_size_16k - self.skip_head
        self.silence_front = extra_frame_16k - (self.window * 5) if self.settings.silenceFront else 0

        # Audio buffer to measure volume between chunks
        audio_buffer_size = block_frame_16k + crossfade_frame_16k
        self.audio_buffer = torch.zeros(audio_buffer_size, dtype=self.dtype, device=self.device_manager.device)

        # Audio buffer for conversion without silence
        self.convert_buffer = torch.zeros(convert_size_16k, dtype=self.dtype, device=self.device_manager.device)
        # Additional +1 is to compensate for pitch extraction algorithm
        # that can output additional feature.
        self.pitch_buffer = torch.zeros(self.convert_feature_size_16k + 1, dtype=torch.int64, device=self.device_manager.device)
        self.pitchf_buffer = torch.zeros(self.convert_feature_size_16k + 1, dtype=self.dtype, device=self.device_manager.device)
        print('[Voice Changer] Allocated audio buffer:', self.audio_buffer.shape[0])
        print('[Voice Changer] Allocated convert buffer:', self.convert_buffer.shape[0])
        print('[Voice Changer] Allocated pitchf buffer:', self.pitchf_buffer.shape[0])

    def convert(self, audio_in: AudioInOutFloat, sample_rate: int) -> torch.Tensor:
        if self.pipeline is None:
            raise PipelineNotInitializedException()

        # Input audio is always float32
        audio_in_t = torch.as_tensor(audio_in, dtype=torch.float32, device=self.device_manager.device)
        if self.is_half:
            audio_in_t = audio_in_t.half()

        convert_feature_size_16k = audio_in_t.shape[0] // self.window

        audio_in_16k = tat.Resample(
            orig_freq=sample_rate,
            new_freq=self.sr,
            dtype=self.dtype
        ).to(self.device_manager.device)(audio_in_t)

        vol_t = torch.sqrt(
            torch.square(audio_in_16k).mean()
        )

        audio_model = self.pipeline.exec(
            self.settings.dstId,
            audio_in_16k,
            None,
            None,
            self.settings.tran,
            self.settings.formantShift,
            self.settings.indexRatio,
            convert_feature_size_16k,
            0,
            self.slotInfo.embOutputLayer,
            self.slotInfo.useFinalProj,
            0,
            convert_feature_size_16k,
            self.settings.protect,
        )

        # TODO: Need to handle resampling for individual files
        # FIXME: Why the heck does it require another sqrt to amplify the volume?
        audio_out: torch.Tensor = self.resampler_out(audio_model * torch.sqrt(vol_t))

        return audio_out

    def inference(self, audio_in: AudioInOutFloat):
        if self.pipeline is None:
            raise PipelineNotInitializedException()

        # Input audio is always float32
        audio_in_t = torch.as_tensor(audio_in, dtype=torch.float32, device=self.device_manager.device)
        if self.is_half:
            audio_in_t = audio_in_t.half()

        audio_in_16k = self.resampler_in(audio_in_t)

        circular_write(audio_in_16k, self.audio_buffer)

        vol_t = torch.sqrt(
            torch.square(self.audio_buffer).mean()
        )
        vol = max(vol_t.item(), 0)

        if vol < self.settings.silentThreshold:
            return None

        circular_write(audio_in_16k, self.convert_buffer)

        audio_model = self.pipeline.exec(
            self.settings.dstId,
            self.convert_buffer,
            self.pitch_buffer,
            self.pitchf_buffer,
            self.settings.tran,
            self.settings.formantShift,
            self.settings.indexRatio,
            self.convert_feature_size_16k,
            self.silence_front,
            self.slotInfo.embOutputLayer,
            self.slotInfo.useFinalProj,
            self.skip_head,
            self.return_length,
            self.settings.protect,
        )

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

        output_file_simple = export2onnx(modelSlot)

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
