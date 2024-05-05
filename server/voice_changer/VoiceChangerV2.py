"""
■ VoiceChangerV2
- VoiceChangerとの差分
・リサンプル処理の無駄を省くため、VoiceChangerModelにリサンプル処理を移譲
・前処理、メイン処理の分割を廃止(VoiceChangeModelでの無駄な型変換などを回避するため)

- 適用VoiceChangerModel
・DiffusionSVC
・RVC
"""

from typing import Any, Union

from const import TMP_DIR
import torch
import os
import numpy as np
from dataclasses import dataclass, asdict, field
from mods.log_control import VoiceChangaerLogger

# from voice_changer.Beatrice.Beatrice import Beatrice

from voice_changer.IORecorder import IORecorder

from voice_changer.utils.Timer import Timer2
from voice_changer.utils.VoiceChangerIF import VoiceChangerIF
from voice_changer.utils.VoiceChangerModel import AudioInOutFloat, VoiceChangerModel
from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    DeviceChangingException,
    HalfPrecisionChangingException,
    NoModeLoadedException,
    NotEnoughDataExtimateF0,
    ONNXInputArgumentException,
    PipelineNotInitializedException,
    VoiceChangerIsNotSelectedException,
)
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.common.deviceManager.DeviceManager import DeviceManager

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")
logger = VoiceChangaerLogger.get_instance().getLogger()


@dataclass
class VoiceChangerV2Settings:
    inputSampleRate: int = 48000  # 48000 or 24000
    outputSampleRate: int = 48000  # 48000 or 24000

    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapSize: int = 4096
    serverReadChunkSize: int = 192
    extraConvertSize: int = 1024 * 4
    gpu: int = -1

    recordIO: int = 0  # 0:off, 1:on

    performance: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # ↓mutableな物だけ列挙
    intData: list[str] = field(
        default_factory=lambda: [
            "inputSampleRate",
            "outputSampleRate",
            "crossFadeOverlapSize",
            "recordIO",
            "serverReadChunkSize",
            "extraConvertSize",
            "gpu"
        ]
    )
    floatData: list[str] = field(
        default_factory=lambda: [
            "crossFadeOffsetRate",
            "crossFadeEndRate",
        ]
    )
    strData: list[str] = field(default_factory=lambda: [])


class VoiceChangerV2(VoiceChangerIF):
    def __init__(self, params: VoiceChangerParams):
        # 初期化
        self.settings = VoiceChangerV2Settings()

        self.block_frame = self.settings.serverReadChunkSize * 128
        self.crossfade_frame = self.settings.crossFadeOverlapSize
        self.input_sample_sec = self.settings.inputSampleRate // 100
        self.sola_search_frame = self.input_sample_sec
        self.sola_buffer_frame = min(self.crossfade_frame, self.input_sample_sec * 4)
        self.extra_frame = self.settings.extraConvertSize

        self.processing_sampling_rate = 0

        self.voiceChangerModel: VoiceChangerModel | None = None
        self.params = params
        self.device_manager = DeviceManager.get_instance()
        self.noCrossFade = False
        self.sola_buffer: torch.Tensor | None = None
        self.ioRecorder: IORecorder | None = None

        logger.info(f"VoiceChangerV2 Initialized")
        np.set_printoptions(threshold=10000)


    def setModel(self, model: VoiceChangerModel):
        self.voiceChangerModel = model
        self.processing_sampling_rate = self.voiceChangerModel.get_processing_sampling_rate()
        self.sola_buffer_frame = min(self.crossfade_frame, self.input_sample_sec * 4)
        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_buffer_frame, self.sola_search_frame)
        self._generate_strength()
        # if model.voiceChangerType == "Beatrice" or model.voiceChangerType == "LLVC":
        if model.voiceChangerType == "Beatrice":
            self.noCrossFade = True
        else:
            self.noCrossFade = False

    def setInputSampleRate(self, sr: int):
        self.settings.inputSampleRate = sr

        self.input_sample_sec = self.settings.inputSampleRate // 100
        self.sola_search_frame = self.input_sample_sec
        self.sola_buffer_frame = min(self.crossfade_frame, self.input_sample_sec * 4)

        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_buffer_frame, self.sola_search_frame)

    def setOutputSampleRate(self, sr: int):
        self.settings.outputSampleRate = sr
        self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)

    def get_info(self):
        data = asdict(self.settings)
        if self.voiceChangerModel is not None:
            data.update(self.voiceChangerModel.get_info())
        return data

    def get_performance(self):
        return self.settings.performance

    def update_settings(self, key: str, val: Any):
        if self.voiceChangerModel is None:
            logger.warn("[Voice Changer] Voice Changer is not selected.")
            return self.get_info()

        if key == "serverAudioStated" and val == 0:
            self.settings.inputSampleRate = 48000
            self.settings.outputSampleRate = 48000
            self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)

        if key in self.settings.intData:
            val = int(val)
            setattr(self.settings, key, val)
            if key == "serverReadChunkSize":
                self.block_frame = self.settings.serverReadChunkSize * 128
                self._generate_strength()
            elif key == 'extraConvertSize':
                self.extra_frame = val
            elif key == 'gpu':
                # When changing GPU, need to re-allocate fade-in/fade-out buffers on different device
                self._generate_strength()
            elif key == 'crossFadeOverlapSize':
                self.crossfade_frame = self.settings.crossFadeOverlapSize
                self.sola_buffer_frame = min(self.crossfade_frame, self.input_sample_sec * 4)
                self._generate_strength()
            elif key == "recordIO" and val == 1:
                if self.ioRecorder is not None:
                    self.ioRecorder.close()
                self.ioRecorder = IORecorder(
                    STREAM_INPUT_FILE,
                    STREAM_OUTPUT_FILE,
                    self.settings.inputSampleRate,
                    self.settings.outputSampleRate,
                    # 16000,
                )
                print(f"-------------------------- - - - {self.settings.inputSampleRate}, {self.settings.outputSampleRate}")
            elif key == "recordIO" and val == 0:
                if self.ioRecorder is not None:
                    self.ioRecorder.close()
                pass
            elif key == "recordIO" and val == 2:
                if self.ioRecorder is not None:
                    self.ioRecorder.close()
            elif key in {"inputSampleRate", "outputSampleRate"}:
                self.voiceChangerModel.setSamplingRate(self.settings.inputSampleRate, self.settings.outputSampleRate)
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
            if key in {"crossFadeOffsetRate", "crossFadeEndRate"}:
                self._generate_strength()
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))

        if key in {'gpu', 'serverReadChunkSize', 'extraConvertSize', 'crossFadeOffsetRate', 'crossFadeEndRate', 'crossFadeOverlapSize'}:
            if self.voiceChangerModel is not None:
                self.voiceChangerModel.realloc(self.block_frame, self.extra_frame, self.crossfade_frame, self.sola_buffer_frame, self.sola_search_frame)

        self.voiceChangerModel.update_settings(key, val)

        return self.get_info()

    def _generate_strength(self):
        cf_offset = int(self.sola_buffer_frame * self.settings.crossFadeOffsetRate)
        cf_end = int(self.sola_buffer_frame * self.settings.crossFadeEndRate)
        cf_range = cf_end - cf_offset
        percent = np.arange(cf_range, dtype=np.float32) / cf_range

        np_prev_strength = np.cos(percent * 0.5 * np.pi) ** 2
        np_cur_strength = np.cos((1 - percent) * 0.5 * np.pi) ** 2

        self.np_prev_strength = np.concatenate(
            [
                np.ones(cf_offset, dtype=np.float32),
                np_prev_strength,
                np.zeros(self.sola_buffer_frame - cf_offset - len(np_prev_strength), dtype=np.float32),
            ]
        )
        self.np_cur_strength = np.concatenate(
            [
                np.zeros(cf_offset, dtype=np.float32),
                np_cur_strength,
                np.ones(self.sola_buffer_frame - cf_offset - len(np_cur_strength), dtype=np.float32),
            ]
        )

        logger.info(f"Generated Strengths: for prev:{self.np_prev_strength.shape}, for cur:{self.np_cur_strength.shape}")

        # ひとつ前の結果とサイズが変わるため、記録は消去する。
        self.sola_buffer = np.zeros(self.sola_buffer_frame, dtype=np.float32)

    def get_processing_sampling_rate(self):
        if self.voiceChangerModel is None:
            return 0
        return self.voiceChangerModel.get_processing_sampling_rate()

    #  audio_in: tuple of short
    def on_request(self, audio_in: AudioInOutFloat) -> tuple[AudioInOutFloat, list[Union[int, float]]]:
        try:
            if self.voiceChangerModel is None:
                raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")

            with Timer2("main-process", True) as t:
                # FIXME: Normally, the client must communicate with exactly the same buffer size.
                # Pre-alloc fixed-size buffer
                audio_in_fixed = np.zeros(self.block_frame, dtype=np.float32)
                # Clamp received data to block frame size
                audio_in = audio_in[-self.block_frame:]
                # Copy into fixed-size buffer
                audio_in_fixed[:audio_in.shape[0]] = audio_in

                if self.noCrossFade:  # Beatrice, LLVC
                    with torch.no_grad():
                        audio = self.voiceChangerModel.inference(
                            audio_in_fixed,
                            crossfade_frame=0,
                            sola_search_frame=0,
                        )
                    # block_frame = audio_in_fixed.shape[0]
                    # result = audio[:block_frame]
                    if audio is None:
                        return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
                    result = audio.detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        # TODO: Maybe audio and sola buffer should be tensors here.
                        audio = self.voiceChangerModel.inference(audio_in_fixed)
                    if audio is None:
                        return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]

                    audio = audio.detach().cpu().numpy()

                    sola_buffer_frame = self.sola_search_frame + self.sola_buffer_frame
                    audio_offset = -(sola_buffer_frame + self.block_frame)
                    audio = audio[audio_offset:]

                    # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC, https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI
                    cor_nom = np.convolve(
                        audio[: sola_buffer_frame],
                        np.flip(self.sola_buffer),
                        "valid",
                    )
                    cor_den = np.sqrt(
                        np.convolve(
                            audio[: sola_buffer_frame] ** 2,
                            np.ones(self.sola_buffer_frame, dtype=np.float32),
                            "valid",
                        )
                        + 0.001
                    )
                    sola_offset = int(np.argmax(cor_nom / cor_den))
                    sola_end = sola_offset + self.block_frame
                    result = audio[sola_offset:sola_end]
                    result[:self.sola_buffer_frame] *= self.np_cur_strength
                    result[:self.sola_buffer_frame] += self.sola_buffer[:]

                    if sola_offset < self.sola_search_frame:
                        offset = -(sola_buffer_frame - sola_offset)
                        end = -(self.sola_search_frame - sola_offset)
                        self.sola_buffer = audio[offset:end] * self.np_prev_strength

            mainprocess_time = t.secs

            # 後処理
            with Timer2("post-process", True) as t:
                print_convert_processing(f" Output data size of {result.shape[0]}/{self.processing_sampling_rate}hz {result.shape[0]}/{self.settings.outputSampleRate}hz")

                # if audio_in_fixed.shape[0] != result.shape[0]:
                # print("TODO FIX:::::PADDING", audio_in_fixed.shape[0], result.shape[0])
                if self.voiceChangerModel.voiceChangerType == "LLVC":
                    outputData = result
                else:
                    outputData = pad_array(result, audio_in_fixed.shape[0])
                # else:
                #     outputData = result

                if self.settings.recordIO == 1:
                    self.ioRecorder.writeInput((audio_in_fixed * 32767).astype(np.int16))
                    self.ioRecorder.writeOutput((outputData * 32767).astype(np.int16).tobytes())

            postprocess_time = t.secs

            print_convert_processing(f" [fin] Input/Output size:{audio_in_fixed.shape[0]},{outputData.shape[0]}")
            perf = [0, mainprocess_time, postprocess_time]

            return outputData, perf

        except NoModeLoadedException as e:
            logger.warn(f"[Voice Changer] [Exception], {e}")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except ONNXInputArgumentException as e:
            logger.warn(f"[Voice Changer] [Exception] onnx are waiting valid input., {e}")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except HalfPrecisionChangingException:
            logger.warn("[Voice Changer] Switching model configuration....")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except NotEnoughDataExtimateF0:
            logger.warn("[Voice Changer] warming up... waiting more data.")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except DeviceChangingException as e:
            logger.warn(f"[Voice Changer] embedder: {e}")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except VoiceChangerIsNotSelectedException:
            logger.warn("[Voice Changer] Voice Changer is not selected. Wait a bit and if there is no improvement, please re-select vc.")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except DeviceCannotSupportHalfPrecisionException:
            # RVC.pyでfallback処理をするので、ここはダミーデータ返すだけ。
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except PipelineNotInitializedException:
            logger.warn("[Voice Changer] Pipeline is not initialized.")
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]
        except Exception as e:
            logger.warn(f"[Voice Changer] VC PROCESSING EXCEPTION!!! {e}")
            logger.exception(e)
            return np.zeros(self.block_frame, dtype=np.float32), [0, 0, 0]

    def export2onnx(self):
        return self.voiceChangerModel.export2onnx()

        ##############

    def merge_models(self, request: str):
        if self.voiceChangerModel is None:
            logger.info("[Voice Changer] Voice Changer is not selected.")
            return
        self.voiceChangerModel.merge_models(request)
        return self.get_info()


PRINT_CONVERT_PROCESSING: bool = False
# PRINT_CONVERT_PROCESSING = True


def print_convert_processing(mess: str):
    if PRINT_CONVERT_PROCESSING:
        logger.info(mess)


def pad_array(arr: AudioInOutFloat, target_length: int):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr

    pad_width = target_length - current_length
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    # padded_arr = np.pad(
    #     arr, (pad_left, pad_right), "constant", constant_values=(0, 0)
    # )
    padded_arr = np.pad(arr, (pad_left, pad_right), "edge")
    return padded_arr
