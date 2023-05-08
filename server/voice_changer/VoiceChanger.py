from typing import Any, Union, cast

import socketio
from const import TMP_DIR, ModelType
import torch
import os
import traceback
import numpy as np
from dataclasses import dataclass, asdict, field
import resampy


from voice_changer.IORecorder import IORecorder
from voice_changer.Local.AudioDeviceList import ServerAudioDevice, list_audio_device
from voice_changer.utils.LoadModelParams import LoadModelParams

from voice_changer.utils.Timer import Timer
from voice_changer.utils.VoiceChangerModel import VoiceChangerModel, AudioInOut
from Exceptions import (
    DeviceChangingException,
    HalfPrecisionChangingException,
    NoModeLoadedException,
    NotEnoughDataExtimateF0,
    ONNXInputArgumentException,
)
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
import pyaudio
import threading
import struct
import time

providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")


@dataclass
class VoiceChangerSettings:
    inputSampleRate: int = 48000  # 48000 or 24000

    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapSize: int = 4096

    recordIO: int = 0  # 0:off, 1:on
    serverAudioInputDevices: list[ServerAudioDevice] = field(default_factory=lambda: [])
    serverAudioOutputDevices: list[ServerAudioDevice] = field(
        default_factory=lambda: []
    )

    enableServerAudio: int = 0  # 0:off, 1:on
    serverAudioStated: int = 0  # 0:off, 1:on
    serverInputAudioSampleRate: int = 48000
    serverOutputAudioSampleRate: int = 48000
    serverInputAudioBufferSize: int = 1024 * 24
    serverOutputAudioBufferSize: int = 1024 * 24
    serverInputDeviceId: int = -1
    serverOutputDeviceId: int = -1
    serverReadChunkSize: int = 256
    performance: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # ↓mutableな物だけ列挙
    intData: list[str] = field(
        default_factory=lambda: [
            "inputSampleRate",
            "crossFadeOverlapSize",
            "recordIO",
            "enableServerAudio",
            "serverAudioStated",
            "serverInputAudioSampleRate",
            "serverOutputAudioSampleRate",
            "serverInputAudioBufferSize",
            "serverOutputAudioBufferSize",
            "serverInputDeviceId",
            "serverOutputDeviceId",
            "serverReadChunkSize",
        ]
    )
    floatData: list[str] = field(
        default_factory=lambda: ["crossFadeOffsetRate", "crossFadeEndRate"]
    )
    strData: list[str] = field(default_factory=lambda: [])


def serverLocal(_vc):
    vc: VoiceChanger = _vc
    audio = pyaudio.PyAudio()

    def createAudioInput(deviceId: int, sampleRate: int, bufferSize: int):
        audio_input_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sampleRate,
            # frames_per_buffer=32768,
            frames_per_buffer=bufferSize,
            input_device_index=deviceId,
            input=True,
        )
        return audio_input_stream

    def createAudioOutput(deviceId: int, sampleRate: int, bufferSize: int):
        audio_output_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sampleRate,
            # frames_per_buffer=32768,
            frames_per_buffer=bufferSize,
            output_device_index=deviceId,
            output=True,
        )
        return audio_output_stream

    currentInputDeviceId = -1
    currentInputSampleRate = -1
    currentInputBufferSize = -1
    currentOutputDeviceId = -1
    currentOutputSampleRate = -1
    currentOutputBufferSize = -1

    audio_input_stream = None
    audio_output_stream = None
    showPerformanceTime = 0
    while True:
        if (
            vc.settings.enableServerAudio == 0
            or vc.settings.serverAudioStated == 0
            or vc.settings.serverInputDeviceId == -1
            or vc.settings.serverOutputDeviceId == -1
        ):
            time.sleep(2)
        else:
            if (
                currentInputDeviceId != vc.settings.serverInputDeviceId
                or currentInputSampleRate != vc.settings.serverInputAudioSampleRate
                or currentInputBufferSize != vc.settings.serverInputAudioBufferSize
            ):
                currentInputDeviceId = vc.settings.serverInputDeviceId
                currentInputSampleRate = vc.settings.serverInputAudioSampleRate
                currentInputBufferSize = vc.settings.serverInputAudioBufferSize
                if audio_input_stream is not None:
                    audio_input_stream.close()
                audio_input_stream = createAudioInput(
                    currentInputDeviceId,
                    currentInputSampleRate,
                    currentInputBufferSize,
                )

            if (
                currentOutputDeviceId != vc.settings.serverOutputDeviceId
                or currentOutputSampleRate != vc.settings.serverOutputAudioSampleRate
                or currentOutputBufferSize != vc.settings.serverOutputAudioBufferSize
            ):
                currentOutputDeviceId = vc.settings.serverOutputDeviceId
                currentOutputSampleRate = vc.settings.serverOutputAudioSampleRate
                currentOutputBufferSize = vc.settings.serverOutputAudioBufferSize
                if audio_output_stream is not None:
                    audio_output_stream.close()
                audio_output_stream = createAudioOutput(
                    currentOutputDeviceId,
                    currentOutputSampleRate,
                    currentOutputBufferSize,
                )
            sampleNum = vc.settings.serverReadChunkSize * 128
            in_wav = audio_input_stream.read(sampleNum, exception_on_overflow=False)
            readNum = len(in_wav) // struct.calcsize("<h")
            unpackedData = np.array(struct.unpack("<%sh" % readNum, in_wav)).astype(
                np.int16
            )
            with Timer("all_inference_time") as t:
                out_wav, times = vc.on_request(unpackedData)
            all_inference_time = t.secs
            performance = [all_inference_time] + times
            performance = [round(x * 1000) for x in performance]
            vc.settings.performance = performance
            currentTime = time.time()
            if currentTime - showPerformanceTime > 5:
                print(sampleNum, readNum, performance)
                showPerformanceTime = currentTime

            audio_output_stream.write(out_wav.tobytes())


class VoiceChanger:
    settings: VoiceChangerSettings
    voiceChanger: VoiceChangerModel
    ioRecorder: IORecorder
    sola_buffer: AudioInOut
    namespace: socketio.AsyncNamespace | None = None

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
        self.mps_enabled: bool = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )

        audioinput, audiooutput = list_audio_device()
        self.settings.serverAudioInputDevices = audioinput
        self.settings.serverAudioOutputDevices = audiooutput

        thread = threading.Thread(target=serverLocal, args=(self,))
        thread.start()
        print(
            f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})"
        )

    def switchModelType(self, modelType: ModelType):
        if hasattr(self, "voiceChanger") and self.voiceChanger is not None:
            # return {"status": "ERROR", "msg": "vc is already selected. currently re-select is not implemented"}
            del self.voiceChanger
            self.voiceChanger = None

        self.modelType = modelType
        if self.modelType == "MMVCv15":
            from voice_changer.MMVCv15.MMVCv15 import MMVCv15

            self.voiceChanger = MMVCv15()  # type: ignore
        elif self.modelType == "MMVCv13":
            from voice_changer.MMVCv13.MMVCv13 import MMVCv13

            self.voiceChanger = MMVCv13()
        elif self.modelType == "so-vits-svc-40v2":
            from voice_changer.SoVitsSvc40v2.SoVitsSvc40v2 import SoVitsSvc40v2

            self.voiceChanger = SoVitsSvc40v2(self.params)
        elif self.modelType == "so-vits-svc-40" or self.modelType == "so-vits-svc-40_c":
            from voice_changer.SoVitsSvc40.SoVitsSvc40 import SoVitsSvc40

            self.voiceChanger = SoVitsSvc40(self.params)
        elif self.modelType == "DDSP-SVC":
            from voice_changer.DDSP_SVC.DDSP_SVC import DDSP_SVC

            self.voiceChanger = DDSP_SVC(self.params)
        elif self.modelType == "RVC":
            from voice_changer.RVC.RVC import RVC

            self.voiceChanger = RVC(self.params)
        else:
            from voice_changer.MMVCv13.MMVCv13 import MMVCv13

            self.voiceChanger = MMVCv13()

        return {"status": "OK", "msg": "vc is switched."}

    def getModelType(self):
        if self.modelType is not None:
            return {"status": "OK", "vc": self.modelType}
        else:
            return {"status": "OK", "vc": "none"}

    def loadModel(self, props: LoadModelParams):
        try:
            return self.voiceChanger.loadModel(props)
        except Exception as e:
            print(traceback.format_exc())
            print("[Voice Changer] Model Load Error! Check your model is valid.", e)
            return {"status": "NG"}

    def get_info(self):
        data = asdict(self.settings)
        if hasattr(self, "voiceChanger"):
            data.update(self.voiceChanger.get_info())
        return data

    def get_performance(self):
        return self.settings.performance

    def update_settings(self, key: str, val: Any):
        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "crossFadeOffsetRate" or key == "crossFadeEndRate":
                self.crossfadeSize = 0
            if key == "recordIO" and val == 1:
                if hasattr(self, "ioRecorder"):
                    self.ioRecorder.close()
                self.ioRecorder = IORecorder(
                    STREAM_INPUT_FILE, STREAM_OUTPUT_FILE, self.settings.inputSampleRate
                )
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
            if hasattr(self, "voiceChanger"):
                ret = self.voiceChanger.update_settings(key, val)
                if ret is False:
                    print(f"{key} is not mutable variable or unknown variable!")
            else:
                print("voice changer is not initialized!")
        return self.get_info()

    def _generate_strength(self, crossfadeSize: int):
        if (
            self.crossfadeSize != crossfadeSize
            or self.currentCrossFadeOffsetRate != self.settings.crossFadeOffsetRate
            or self.currentCrossFadeEndRate != self.settings.crossFadeEndRate
            or self.currentCrossFadeOverlapSize != self.settings.crossFadeOverlapSize
        ):
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

            print(
                f"Generated Strengths: for prev:{self.np_prev_strength.shape}, for cur:{self.np_cur_strength.shape}"
            )

            # ひとつ前の結果とサイズが変わるため、記録は消去する。
            if hasattr(self, "np_prev_audio1") is True:
                delattr(self, "np_prev_audio1")
            if hasattr(self, "sola_buffer") is True:
                del self.sola_buffer

    #  receivedData: tuple of short
    def on_request(
        self, receivedData: AudioInOut
    ) -> tuple[AudioInOut, list[Union[int, float]]]:
        return self.on_request_sola(receivedData)

    def on_request_sola(
        self, receivedData: AudioInOut
    ) -> tuple[AudioInOut, list[Union[int, float]]]:
        try:
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

                data = self.voiceChanger.generate_input(
                    newData, block_frame, crossfade_frame, sola_search_frame
                )
            preprocess_time = t.secs

            # 変換処理
            with Timer("main-process") as t:
                # Inference
                audio = self.voiceChanger.inference(data)

                if hasattr(self, "sola_buffer") is True:
                    np.set_printoptions(threshold=10000)
                    audio_offset = -1 * (
                        sola_search_frame + crossfade_frame + block_frame
                    )
                    audio = audio[audio_offset:]
                    a = 0
                    audio = audio[a:]
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
                    print("[Voice Changer] no sola buffer. (You can ignore this.)")
                    result = np.zeros(4096).astype(np.int16)

                if (
                    hasattr(self, "sola_buffer") is True
                    and sola_offset < sola_search_frame
                ):
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

                print_convert_processing(
                    f" Output data size of {result.shape[0]}/{processing_sampling_rate}hz {outputData.shape[0]}/{self.settings.inputSampleRate}hz"
                )

                if self.settings.recordIO == 1:
                    self.ioRecorder.writeInput(receivedData)
                    self.ioRecorder.writeOutput(outputData.tobytes())

                # if receivedData.shape[0] != outputData.shape[0]:
                #     print(f"Padding, in:{receivedData.shape[0]} out:{outputData.shape[0]}")
                #     outputData = pad_array(outputData, receivedData.shape[0])
                #     # print_convert_processing(
                #     #     f" Padded!, Output data size of {result.shape[0]}/{processing_sampling_rate}hz {outputData.shape[0]}/{self.settings.inputSampleRate}hz")
            postprocess_time = t.secs

            print_convert_processing(
                f" [fin] Input/Output size:{receivedData.shape[0]},{outputData.shape[0]}"
            )
            perf = [preprocess_time, mainprocess_time, postprocess_time]
            return outputData, perf

        except NoModeLoadedException as e:
            print("[Voice Changer] [Exception]", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except ONNXInputArgumentException as e:
            print("[Voice Changer] [Exception]", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except HalfPrecisionChangingException as e:
            print("[Voice Changer] Switching model configuration....", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except NotEnoughDataExtimateF0 as e:
            print("[Voice Changer] not enough data", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except DeviceChangingException as e:
            print("[Voice Changer] embedder:", e)
            return np.zeros(1).astype(np.int16), [0, 0, 0]
        except Exception as e:
            print("VC PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            return np.zeros(1).astype(np.int16), [0, 0, 0]

    def export2onnx(self):
        return self.voiceChanger.export2onnx()

        ##############

    def merge_models(self, request: str):
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
        padded_arr = np.pad(
            arr, (pad_left, pad_right), "constant", constant_values=(0, 0)
        )
        return padded_arr
