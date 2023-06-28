from typing import Any, Union, cast

from const import TMP_DIR, ModelType
import torch
import os
import traceback
import numpy as np
from dataclasses import dataclass, asdict, field
import resampy


from voice_changer.IORecorder import IORecorder
from voice_changer.utils.LoadModelParams import LoadModelParams

from voice_changer.utils.Timer import Timer
from voice_changer.utils.VoiceChangerModel import AudioInOut
from Exceptions import (
    DeviceCannotSupportHalfPrecisionException,
    DeviceChangingException,
    HalfPrecisionChangingException,
    NoModeLoadedException,
    NotEnoughDataExtimateF0,
    ONNXInputArgumentException,
    VoiceChangerIsNotSelectedException,
)
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")


@dataclass
class VoiceChangerSettings:
    inputSampleRate: int = 48000  # 48000 or 24000

    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapSize: int = 4096

    recordIO: int = 0  # 0:off, 1:on

    performance: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # ↓mutableな物だけ列挙
    intData: list[str] = field(
        default_factory=lambda: [
            "inputSampleRate",
            "crossFadeOverlapSize",
            "recordIO",
        ]
    )
    floatData: list[str] = field(
        default_factory=lambda: [
            "crossFadeOffsetRate",
            "crossFadeEndRate",
        ]
    )
    strData: list[str] = field(default_factory=lambda: [])


class VoiceChanger:
    ioRecorder: IORecorder
    sola_buffer: AudioInOut

    def __init__(self, params: VoiceChangerParams):
        # 初期化
        self.settings = VoiceChangerSettings()
        self.onnx_session = None
        self.currentCrossFadeOffsetRate = 0.0
        self.currentCrossFadeEndRate = 0.0
        self.currentCrossFadeOverlapSize = 0  # setting
        self.crossfadeSize = 0  # calculated

        self.voiceChanger = None
        self.modelType: ModelType | None = None
        self.params = params
        self.gpu_num = torch.cuda.device_count()
        self.prev_audio = np.zeros(4096)
        self.mps_enabled: bool = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        print(f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})")

    def setModel(self, model: Any):
        self.voiceChanger = model

    def getModelType(self):
        if self.modelType is not None:
            return {"status": "OK", "vc": self.modelType}
        else:
            return {"status": "OK", "vc": "none"}

    def loadModel(self, props: LoadModelParams):
        try:
            if self.voiceChanger is None:
                raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")
            return self.voiceChanger.loadModel(props)
        except Exception as e:
            print(traceback.format_exc())
            print("[Voice Changer] Model Load Error! Check your model is valid.", e)
            return {"status": "NG"}

    def get_info(self):
        data = asdict(self.settings)
        if self.voiceChanger is not None:
            data.update(self.voiceChanger.get_info())
        return data

    def get_performance(self):
        return self.settings.performance

    def update_settings(self, key: str, val: Any):
        if self.voiceChanger is None:
            print("[Voice Changer] Voice Changer is not selected.")
            return self.get_info()

        if key == "serverAudioStated" and val == 0:
            self.settings.inputSampleRate = 48000

        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "crossFadeOffsetRate" or key == "crossFadeEndRate":
                self.crossfadeSize = 0
            if key == "recordIO" and val == 1:
                if hasattr(self, "ioRecorder"):
                    self.ioRecorder.close()
                self.ioRecorder = IORecorder(STREAM_INPUT_FILE, STREAM_OUTPUT_FILE, self.settings.inputSampleRate)
            if key == "recordIO" and val == 0:
                if hasattr(self, "ioRecorder"):
                    self.ioRecorder.close()
                pass
            if key == "recordIO" and val == 2:
                if hasattr(self, "ioRecorder"):
                    self.ioRecorder.close()

        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            ret = self.voiceChanger.update_settings(key, val)
            if ret is False:
                pass
                # print(f"({key} is not mutable variable or unknown variable)")
        return self.get_info()

    def _generate_strength(self, crossfadeSize: int):
        if self.crossfadeSize != crossfadeSize or self.currentCrossFadeOffsetRate != self.settings.crossFadeOffsetRate or self.currentCrossFadeEndRate != self.settings.crossFadeEndRate or self.currentCrossFadeOverlapSize != self.settings.crossFadeOverlapSize:
            self.crossfadeSize = crossfadeSize
            self.currentCrossFadeOffsetRate = self.settings.crossFadeOffsetRate
            self.currentCrossFadeEndRate = self.settings.crossFadeEndRate
            self.currentCrossFadeOverlapSize = self.settings.crossFadeOverlapSize

            cf_offset = int(crossfadeSize * self.settings.crossFadeOffsetRate)
            cf_end = int(crossfadeSize * self.settings.crossFadeEndRate)
            cf_range = cf_end - cf_offset
            percent = np.arange(cf_range) / cf_range

            np_prev_strength = np.cos(percent * 0.5 * np.pi) ** 2
            np_cur_strength = np.cos((1 - percent) * 0.5 * np.pi) ** 2

            self.np_prev_strength = np.concatenate(
                [
                    np.ones(cf_offset),
                    np_prev_strength,
                    np.zeros(crossfadeSize - cf_offset - len(np_prev_strength)),
                ]
            )
            self.np_cur_strength = np.concatenate(
                [
                    np.zeros(cf_offset),
                    np_cur_strength,
                    np.ones(crossfadeSize - cf_offset - len(np_cur_strength)),
                ]
            )

            print(f"Generated Strengths: for prev:{self.np_prev_strength.shape}, for cur:{self.np_cur_strength.shape}")

            # ひとつ前の結果とサイズが変わるため、記録は消去する。
            if hasattr(self, "np_prev_audio1") is True:
                delattr(self, "np_prev_audio1")
            if hasattr(self, "sola_buffer") is True:
                del self.sola_buffer

    def get_processing_sampling_rate(self):
        if self.voiceChanger is None:
            return 0
        else:
            return self.voiceChanger.get_processing_sampling_rate()

    #  receivedData: tuple of short
    def on_request(self, receivedData: AudioInOut) -> tuple[AudioInOut, list[Union[int, float]]]:
        return self.on_request_sola(receivedData)

    def on_request_sola(self, receivedData: AudioInOut) -> tuple[AudioInOut, list[Union[int, float]]]:
        try:
            if self.voiceChanger is None:
                raise VoiceChangerIsNotSelectedException("Voice Changer is not selected.")

            processing_sampling_rate = self.voiceChanger.get_processing_sampling_rate()
            # 前処理
            with Timer("pre-process") as t:
                if self.settings.inputSampleRate != processing_sampling_rate:
                    newData = cast(
                        AudioInOut,
                        resampy.resample(
                            receivedData,
                            self.settings.inputSampleRate,
                            processing_sampling_rate,
                        ),
                    )
                else:
                    newData = receivedData

                sola_search_frame = int(0.012 * processing_sampling_rate)
                # sola_search_frame = 0
                block_frame = newData.shape[0]
                crossfade_frame = min(self.settings.crossFadeOverlapSize, block_frame)
                self._generate_strength(crossfade_frame)

                data = self.voiceChanger.generate_input(newData, block_frame, crossfade_frame, sola_search_frame)
            preprocess_time = t.secs

            # 変換処理
            with Timer("main-process") as t:
                # Inference
                audio = self.voiceChanger.inference(data)

                if hasattr(self, "sola_buffer") is True:
                    np.set_printoptions(threshold=10000)
                    audio_offset = -1 * (sola_search_frame + crossfade_frame + block_frame)
                    audio = audio[audio_offset:]

                    # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC, https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI
                    cor_nom = np.convolve(
                        audio[: crossfade_frame + sola_search_frame],
                        np.flip(self.sola_buffer),
                        "valid",
                    )
                    cor_den = np.sqrt(
                        np.convolve(
                            audio[: crossfade_frame + sola_search_frame] ** 2,
                            np.ones(crossfade_frame),
                            "valid",
                        )
                        + 1e-3
                    )
                    sola_offset = int(np.argmax(cor_nom / cor_den))
                    sola_end = sola_offset + block_frame
                    output_wav = audio[sola_offset:sola_end].astype(np.float64)
                    output_wav[:crossfade_frame] *= self.np_cur_strength
                    output_wav[:crossfade_frame] += self.sola_buffer[:]

                    result = output_wav
                else:
                    print("[Voice Changer] warming up... generating sola buffer.")
                    result = np.zeros(4096).astype(np.int16)

                if hasattr(self, "sola_buffer") is True and sola_offset < sola_search_frame:
                    offset = -1 * (sola_search_frame + crossfade_frame - sola_offset)
                    end = -1 * (sola_search_frame - sola_offset)
                    sola_buf_org = audio[offset:end]
                    self.sola_buffer = sola_buf_org * self.np_prev_strength
                else:
                    self.sola_buffer = audio[-crossfade_frame:] * self.np_prev_strength
                    # self.sola_buffer = audio[- crossfade_frame:]
            mainprocess_time = t.secs

            # 後処理
            with Timer("post-process") as t:
                result = result.astype(np.int16)
                if self.settings.inputSampleRate != processing_sampling_rate:
                    # print(
                    #     "samplingrate",
                    #     self.settings.inputSampleRate,
                    #     processing_sampling_rate,
                    # )
                    outputData = cast(
                        AudioInOut,
                        resampy.resample(
                            result,
                            processing_sampling_rate,
                            self.settings.inputSampleRate,
                        ).astype(np.int16),
                    )
                else:
                    outputData = result

                print_convert_processing(f" Output data size of {result.shape[0]}/{processing_sampling_rate}hz {outputData.shape[0]}/{self.settings.inputSampleRate}hz")

                if receivedData.shape[0] != outputData.shape[0]:
                    # print(
                    #     f"Padding, in:{receivedData.shape[0]} out:{outputData.shape[0]}"
                    # )
                    outputData = pad_array(outputData, receivedData.shape[0])
                    # print_convert_processing(
                    #     f" Padded!, Output data size of {result.shape[0]}/{processing_sampling_rate}hz {outputData.shape[0]}/{self.settings.inputSampleRate}hz")
                    pass

                if self.settings.recordIO == 1:
                    self.ioRecorder.writeInput(receivedData)
                    self.ioRecorder.writeOutput(outputData.tobytes())

            postprocess_time = t.secs

            print_convert_processing(f" [fin] Input/Output size:{receivedData.shape[0]},{outputData.shape[0]}")
            perf = [preprocess_time, mainprocess_time, postprocess_time]
            return outputData, perf

        except NoModeLoadedException as e:
            print("[Voice Changer] [Exception]", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except ONNXInputArgumentException as e:
            print("[Voice Changer] [Exception] onnx are waiting valid input.", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except HalfPrecisionChangingException:
            print("[Voice Changer] Switching model configuration....")
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except NotEnoughDataExtimateF0:
            print("[Voice Changer] warming up... waiting more data.")
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except DeviceChangingException as e:
            print("[Voice Changer] embedder:", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except VoiceChangerIsNotSelectedException:
            print("[Voice Changer] Voice Changer is not selected. Wait a bit and if there is no improvement, please re-select vc.")
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except DeviceCannotSupportHalfPrecisionException:
            # RVC.pyでfallback処理をするので、ここはダミーデータ返すだけ。
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except Exception as e:
            print("[Voice Changer] VC PROCESSING EXCEPTION!!!", e)
            print(traceback.format_exc())
            return np.zeros(1).astype(np.int16), [0, 0, 0]

    def export2onnx(self):
        return self.voiceChanger.export2onnx()

        ##############

    def merge_models(self, request: str):
        if self.voiceChanger is None:
            print("[Voice Changer] Voice Changer is not selected.")
            return
        self.voiceChanger.merge_models(request)
        return self.get_info()


PRINT_CONVERT_PROCESSING: bool = False
# PRINT_CONVERT_PROCESSING = True


def print_convert_processing(mess: str):
    if PRINT_CONVERT_PROCESSING is True:
        print(mess)


def pad_array(arr: AudioInOut, target_length: int):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        # padded_arr = np.pad(
        #     arr, (pad_left, pad_right), "constant", constant_values=(0, 0)
        # )
        padded_arr = np.pad(arr, (pad_left, pad_right), "edge")
        return padded_arr
