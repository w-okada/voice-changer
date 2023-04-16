from typing import Any, Callable, Optional, Protocol, TypeAlias, Union, cast
from const import TMP_DIR, ModelType
import torch
import os
import traceback
import numpy as np
from dataclasses import dataclass, asdict, field
import resampy


from voice_changer.IORecorder import IORecorder
# from voice_changer.IOAnalyzer import IOAnalyzer

from voice_changer.utils.Timer import Timer
from voice_changer.utils.VoiceChangerModel import VoiceChangerModel, AudioInOut
import time


providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")
STREAM_ANALYZE_FILE_DIO = os.path.join(TMP_DIR, "analyze-dio.png")
STREAM_ANALYZE_FILE_HARVEST = os.path.join(TMP_DIR, "analyze-harvest.png")


@dataclass
class VoiceChangerSettings():
    inputSampleRate: int = 24000  # 48000 or 24000

    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapSize: int = 4096
    solaEnabled: int = 1  # 0:off, 1:on

    recordIO: int = 0  # 0:off, 1:on

    # ↓mutableな物だけ列挙
    intData: list[str] = field(
        default_factory=lambda: ["inputSampleRate", "crossFadeOverlapSize", "recordIO", "solaEnabled"]
    )
    floatData: list[str] = field(
        default_factory=lambda: ["crossFadeOffsetRate", "crossFadeEndRate"]
    )
    strData: list[str] = field(
        default_factory=lambda: []
    )


class VoiceChanger():
    settings: VoiceChangerSettings
    voiceChanger: VoiceChangerModel

    def __init__(self, params):
        # 初期化
        self.settings = VoiceChangerSettings()
        self.onnx_session = None
        self.currentCrossFadeOffsetRate = 0
        self.currentCrossFadeEndRate = 0
        self.currentCrossFadeOverlapSize = 0  # setting
        self.crossfadeSize = 0  # calculated

        self.voiceChanger = None
        self.modelType = None
        self.params = params
        self.gpu_num = torch.cuda.device_count()
        self.prev_audio = np.zeros(4096)
        self.mps_enabled: bool = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        print(f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})")

    def switchModelType(self, modelType: ModelType):
        if hasattr(self, "voiceChanger") and self.voiceChanger != None:
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
        if self.modelType != None:
            return {"status": "OK", "vc": self.modelType}
        else:
            return {"status": "OK", "vc": "none"}

    def loadModel(
        self,
        props,
    ):

        try:
            return self.voiceChanger.loadModel(props)
        except Exception as e:
            print("[Voice Changer] Model Load Error! Check your model is valid.", e)
            return {"status": "NG"}

        # try:
        #     if self.modelType == "MMVCv15" or self.modelType == "MMVCv13":
        #         return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file)
        #     elif self.modelType == "so-vits-svc-40" or self.modelType == "so-vits-svc-40_c" or self.modelType == "so-vits-svc-40v2":
        #         return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file, clusterTorchModel)
        #     elif self.modelType == "RVC":
        #         return self.voiceChanger.loadModel(slot, config, pyTorch_model_file, onnx_model_file, feature_file, index_file, is_half)
        #     else:
        #         return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file, clusterTorchModel)
        # except Exception as e:
        #     print("[Voice Changer] Model Load Error! Check your model is valid.", e)
        #     return {"status": "NG"}

    def get_info(self):
        data = asdict(self.settings)
        if hasattr(self, "voiceChanger"):
            data.update(self.voiceChanger.get_info())
        return data

    def update_settings(self, key: str, val: Any):
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

                # if hasattr(self, "ioAnalyzer") == False:
                #     self.ioAnalyzer = IOAnalyzer()

                # try:
                #     self.ioAnalyzer.analyze(STREAM_INPUT_FILE, STREAM_ANALYZE_FILE_DIO, STREAM_ANALYZE_FILE_HARVEST, self.settings.inputSampleRate)

                # except Exception as e:
                #     print("recordIO exception", e)
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            if hasattr(self, "voiceChanger"):
                ret = self.voiceChanger.update_settings(key, val)
                if ret == False:
                    print(f"{key} is not mutable variable or unknown variable!")
            else:
                print(f"voice changer is not initialized!")
        return self.get_info()

    def _generate_strength(self, crossfadeSize: int):

        if self.crossfadeSize != crossfadeSize or \
                self.currentCrossFadeOffsetRate != self.settings.crossFadeOffsetRate or \
                self.currentCrossFadeEndRate != self.settings.crossFadeEndRate or \
                self.currentCrossFadeOverlapSize != self.settings.crossFadeOverlapSize:

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

            self.np_prev_strength = np.concatenate([np.ones(cf_offset), np_prev_strength,
                                                   np.zeros(crossfadeSize - cf_offset - len(np_prev_strength))])
            self.np_cur_strength = np.concatenate([np.zeros(cf_offset), np_cur_strength, np.ones(crossfadeSize - cf_offset - len(np_cur_strength))])

            print(f"Generated Strengths: for prev:{self.np_prev_strength.shape}, for cur:{self.np_cur_strength.shape}")

            # ひとつ前の結果とサイズが変わるため、記録は消去する。
            if hasattr(self, 'np_prev_audio1') == True:
                delattr(self, "np_prev_audio1")
            if hasattr(self, "sola_buffer"):
                del self.sola_buffer

    #  receivedData: tuple of short
    def on_request(self, receivedData: AudioInOut) -> tuple[AudioInOut, list[Union[int, float]]]:
        return self.on_request_sola(receivedData)

    def on_request_sola(self, receivedData: AudioInOut) -> tuple[AudioInOut, list[Union[int, float]]]:
        processing_sampling_rate = self.voiceChanger.get_processing_sampling_rate()

        # 前処理
        with Timer("pre-process") as t:
            if self.settings.inputSampleRate != processing_sampling_rate:
                newData = cast(AudioInOut, resampy.resample(receivedData, self.settings.inputSampleRate, processing_sampling_rate))
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
            try:
                # Inference
                audio = self.voiceChanger.inference(data)

                if hasattr(self, 'sola_buffer') == True:
                    np.set_printoptions(threshold=10000)
                    audio = audio[-sola_search_frame - crossfade_frame - block_frame:]
                    # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC, https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI
                    cor_nom = np.convolve(audio[: crossfade_frame + sola_search_frame], np.flip(self.sola_buffer), 'valid')
                    cor_den = np.sqrt(np.convolve(audio[: crossfade_frame + sola_search_frame] ** 2, np.ones(crossfade_frame), 'valid') + 1e-3)
                    sola_offset = np.argmax(cor_nom / cor_den)

                    output_wav = audio[sola_offset: sola_offset + block_frame].astype(np.float64)
                    output_wav[:crossfade_frame] *= self.np_cur_strength
                    output_wav[:crossfade_frame] += self.sola_buffer[:]

                    result = output_wav
                else:
                    print("no sola buffer")
                    result = np.zeros(4096).astype(np.int16)

                if hasattr(self, 'sola_buffer') == True and sola_offset < sola_search_frame:
                    sola_buf_org = audio[- sola_search_frame - crossfade_frame + sola_offset: -sola_search_frame + sola_offset]
                    self.sola_buffer = sola_buf_org * self.np_prev_strength
                else:
                    self.sola_buffer = audio[- crossfade_frame:] * self.np_prev_strength
                    # self.sola_buffer = audio[- crossfade_frame:]

            except Exception as e:
                print("VC PROCESSING!!!! EXCEPTION!!!", e)
                print(traceback.format_exc())
                return np.zeros(1).astype(np.int16), [0, 0, 0]
        mainprocess_time = t.secs

        # 後処理
        with Timer("post-process") as t:
            result = result.astype(np.int16)
            if self.settings.inputSampleRate != processing_sampling_rate:
                outputData = cast(AudioInOut, resampy.resample(result, processing_sampling_rate, self.settings.inputSampleRate).astype(np.int16))
            else:
                outputData = result

            print_convert_processing(
                f" Output data size of {result.shape[0]}/{processing_sampling_rate}hz {outputData.shape[0]}/{self.settings.inputSampleRate}hz")

            if self.settings.recordIO == 1:
                self.ioRecorder.writeInput(receivedData)
                self.ioRecorder.writeOutput(outputData.tobytes())

            # if receivedData.shape[0] != outputData.shape[0]:
            #     print(f"Padding, in:{receivedData.shape[0]} out:{outputData.shape[0]}")
            #     outputData = pad_array(outputData, receivedData.shape[0])
            #     # print_convert_processing(
            #     #     f" Padded!, Output data size of {result.shape[0]}/{processing_sampling_rate}hz {outputData.shape[0]}/{self.settings.inputSampleRate}hz")
        postprocess_time = t.secs

        print_convert_processing(f" [fin] Input/Output size:{receivedData.shape[0]},{outputData.shape[0]}")
        perf = [preprocess_time, mainprocess_time, postprocess_time]
        return outputData, perf

    def export2onnx(self):
        return self.voiceChanger.export2onnx()


        ##############
PRINT_CONVERT_PROCESSING: bool = False
# PRINT_CONVERT_PROCESSING = True


def print_convert_processing(mess: str):
    if PRINT_CONVERT_PROCESSING == True:
        print(mess)


def pad_array(arr: AudioInOut, target_length: int):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr
