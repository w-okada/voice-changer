"""

"""


from dataclasses import asdict
from typing import Union
import os
import numpy as np
from const import MODEL_DIR_STATIC
from data.ModelSlot import BeatriceModelSlot
from mods.log_control import VoiceChangaerLogger
from voice_changer.Beatrice.BeatriceSettings import BeatriceSettings

from voice_changer.utils.VoiceChangerModel import AudioInOut, VoiceChangerModel
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

from beatrice_internal_api import BeatriceInternalAPI

logger = VoiceChangaerLogger.get_instance().getLogger()


class BeatriceAPI(BeatriceInternalAPI):
    def __init__(self, sample_rate: float = 48000.0):
        if sample_rate < 1000.0:
            raise ValueError(sample_rate)
        super().__init__(float(sample_rate))

    def get_n_speakers(self):
        return 500

    def get_target_speaker_names(self):
        names = []
        for i in range(1, 101):
            names.append(f"[商用不可] jvs{i:03d}")
            names.append(f"[商用不可] jvs{i:03d} -1")
            names.append(f"[商用不可] jvs{i:03d} -2")
            names.append(f"[商用不可] jvs{i:03d} +1")
            names.append(f"[商用不可] jvs{i:03d} +2")
        return names

    def set_sample_rate(self, sample_rate: float):
        if sample_rate < 1000.0:
            raise ValueError(sample_rate)
        super().set_sample_rate(float(sample_rate))

    def set_target_speaker_id(self, target_speaker_id: int):
        if not 0 <= target_speaker_id < self.get_n_speakers():
            raise ValueError(target_speaker_id)
        super().set_target_speaker_id(int(target_speaker_id))

    def read_parameters(self, filename: Union[str, bytes, os.PathLike]):
        super().read_parameters(filename)

    def convert(self, in_wav: np.ndarray) -> np.ndarray:
        if in_wav.ndim != 1:
            raise ValueError(in_wav.ndim)
        if in_wav.dtype != np.float32:
            raise ValueError(in_wav.dtype)
        out_wav = super().convert(in_wav)
        assert in_wav.shape == out_wav.shape
        return out_wav


class Beatrice(VoiceChangerModel):
    def __init__(self, params: VoiceChangerParams, slotInfo: BeatriceModelSlot, static: bool = False):
        logger.info("[Voice Changer] [Beatrice] Creating instance ")
        self.settings = BeatriceSettings()
        self.params = params

        self.prevVol = 0.0
        self.slotInfo = slotInfo
        self.audio_buffer: AudioInOut | None = None

        self.static = static

    def initialize(self):
        logger.info("[Voice Changer] [Beatrice] Initializing... ")

        self.beatrice_api = BeatriceAPI()
        if self.static:
            modelPath = os.path.join(MODEL_DIR_STATIC, str(self.slotInfo.slotIndex), os.path.basename(self.slotInfo.modelFile))
        else:
            modelPath = os.path.join(self.params.model_dir, str(self.slotInfo.slotIndex), os.path.basename(self.slotInfo.modelFile))
        self.beatrice_api.read_parameters(modelPath)
        self.beatrice_api.set_sample_rate(self.inputSampleRate)

        # その他の設定
        self.settings.dstId = self.slotInfo.dstId
        logger.info("[Voice Changer] [Beatrice] Initializing... done")

    def setSamplingRate(self, inputSampleRate, outputSampleRate):
        if inputSampleRate == outputSampleRate:
            self.inputSampleRate = inputSampleRate
            self.outputSampleRate = outputSampleRate
            self.initialize()
        else:
            print("inputSampleRate, outputSampleRate", inputSampleRate, outputSampleRate)

    def update_settings(self, key: str, val: int | float | str):
        logger.info(f"[Voice Changer][Beatrice]: update_settings {key}:{val}")
        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            return False
        return True

    def get_info(self):
        data = asdict(self.settings)
        return data

    def get_processing_sampling_rate(self):
        return self.inputSampleRate

    def generate_input(
        self,
        newData: AudioInOut,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / 32768.0
        # 過去のデータに連結
        if self.audio_buffer is not None:
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)
        else:
            self.audio_buffer = newData

        convertSize = newData.shape[0] + crossfadeSize + solaSearchFrame

        # バッファがたまっていない場合はzeroで補う
        if self.audio_buffer.shape[0] < convertSize:
            self.audio_buffer = np.concatenate([np.zeros([convertSize]), self.audio_buffer])

        # 変換対象の部分だけ抽出
        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]

        return (self.audio_buffer,)

    def inference(self, receivedData: AudioInOut, crossfade_frame: int, sola_search_frame: int):
        data = self.generate_input(receivedData, crossfade_frame, sola_search_frame)
        audio = (data[0]).astype(np.float32)

        self.beatrice_api.set_target_speaker_id(self.settings.dstId)

        block_size = 500
        out_wav_blocks = []
        head = 0
        while head < len(audio):
            in_wav_block = audio[head : head + block_size]
            out_wav_block = self.beatrice_api.convert(in_wav_block)
            out_wav_blocks.append(out_wav_block)
            head += block_size
        out_wav = np.concatenate(out_wav_blocks)
        assert audio.shape == out_wav.shape

        return (out_wav * 32767.0).astype(np.int16)

    def __del__(self):
        del self.pipeline

    # def export2onnx(self):
    #     modelSlot = self.slotInfo

    #     if modelSlot.isONNX:
    #         print("[Voice Changer] export2onnx, No pyTorch filepath.")
    #         return {"status": "ng", "path": ""}

    #     output_file_simple = export2onnx(self.settings.gpu, modelSlot)
    #     return {
    #         "status": "ok",
    #         "path": f"/tmp/{output_file_simple}",
    #         "filename": output_file_simple,
    #     }

    def get_model_current(self):
        return [
            {
                "key": "dstId",
                "val": self.settings.dstId,
            },
        ]
