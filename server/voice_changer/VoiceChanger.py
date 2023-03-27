from const import TMP_DIR, getModelType
import torch
import os
import traceback
import numpy as np
from dataclasses import dataclass, asdict
import resampy


from voice_changer.IORecorder import IORecorder
# from voice_changer.IOAnalyzer import IOAnalyzer


import time
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")
STREAM_ANALYZE_FILE_DIO = os.path.join(TMP_DIR, "analyze-dio.png")
STREAM_ANALYZE_FILE_HARVEST = os.path.join(TMP_DIR, "analyze-harvest.png")


@dataclass
class VocieChangerSettings():
    inputSampleRate: int = 24000  # 48000 or 24000

    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapSize: int = 4096

    recordIO: int = 0  # 0:off, 1:on

    # ↓mutableな物だけ列挙
    intData = ["inputSampleRate", "crossFadeOverlapSize", "recordIO"]
    floatData = ["crossFadeOffsetRate", "crossFadeEndRate"]
    strData = []


class VoiceChanger():

    def __init__(self, params):
        # 初期化
        self.settings = VocieChangerSettings()
        self.onnx_session = None
        self.currentCrossFadeOffsetRate = 0
        self.currentCrossFadeEndRate = 0
        self.currentCrossFadeOverlapSize = 0  # setting
        self.crossfadeSize = 0  # calculated

        self.modelType = getModelType()
        print("[VoiceChanger] activate model type:", self.modelType)
        if self.modelType == "MMVCv15":
            from voice_changer.MMVCv15.MMVCv15 import MMVCv15
            self.voiceChanger = MMVCv15()
        elif self.modelType == "MMVCv13":
            from voice_changer.MMVCv13.MMVCv13 import MMVCv13
            self.voiceChanger = MMVCv13()
        elif self.modelType == "so-vits-svc-40v2" or self.modelType == "so-vits-svc-40v2_c":
            from voice_changer.SoVitsSvc40v2.SoVitsSvc40v2 import SoVitsSvc40v2
            self.voiceChanger = SoVitsSvc40v2(params)
        elif self.modelType == "so-vits-svc-40":
            from voice_changer.SoVitsSvc40.SoVitsSvc40 import SoVitsSvc40
            self.voiceChanger = SoVitsSvc40(params)
        elif self.modelType == "DDSP-SVC":
            from voice_changer.DDSP_SVC.DDSP_SVC import DDSP_SVC
            self.voiceChanger = DDSP_SVC(params)

        else:
            from voice_changer.MMVCv13.MMVCv13 import MMVCv13
            self.voiceChanger = MMVCv13()

        self.gpu_num = torch.cuda.device_count()
        self.prev_audio = np.zeros(4096)
        self.mps_enabled = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        print(f"VoiceChanger Initialized (GPU_NUM:{self.gpu_num}, mps_enabled:{self.mps_enabled})")

    def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None, clusterTorchModel: str = None):
        if self.modelType == "MMVCv15" or self.modelType == "MMVCv13":
            return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file)
        elif self.modelType == "so-vits-svc-40" or self.modelType == "so-vits-svc-40v2" or self.modelType == "so-vits-svc-40v2_c":
            return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file, clusterTorchModel)
        else:
            return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file, clusterTorchModel)

    def get_info(self):
        data = asdict(self.settings)
        data.update(self.voiceChanger.get_info())
        return data

    def update_setteings(self, key: str, val: any):
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
            ret = self.voiceChanger.update_setteings(key, val)
            if ret == False:
                print(f"{key} is not mutable variable or unknown variable!")

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

    #  receivedData: tuple of short
    def on_request(self, receivedData: any):
        processing_sampling_rate = self.voiceChanger.get_processing_sampling_rate()

        print_convert_processing(f"------------ Convert processing.... ------------")
        # 前処理
        with Timer("pre-process") as t:

            with Timer("pre-process") as t1:

                if self.settings.inputSampleRate != processing_sampling_rate:
                    newData = resampy.resample(receivedData, self.settings.inputSampleRate, processing_sampling_rate)
                else:
                    newData = receivedData
            # print("t1::::", t1.secs)
            inputSize = newData.shape[0]
            crossfadeSize = min(self.settings.crossFadeOverlapSize, inputSize)

            print_convert_processing(
                f" Input data size: {receivedData.shape[0]}/{self.settings.inputSampleRate}hz {inputSize}/{processing_sampling_rate}hz")
            print_convert_processing(
                f" Crossfade data size: crossfade:{crossfadeSize}, crossfade setting:{self.settings.crossFadeOverlapSize}, input size:{inputSize}")

            print_convert_processing(f" Convert data size of {inputSize + crossfadeSize} (+ extra size)")
            print_convert_processing(f"         will be cropped:{-1 * (inputSize + crossfadeSize)}, {-1 * (crossfadeSize)}")

            self._generate_strength(crossfadeSize)
            with Timer("pre-process") as t2:
                data = self.voiceChanger.generate_input(newData, inputSize, crossfadeSize)
            # print("t2::::", t2.secs)
        preprocess_time = t.secs

        # 変換処理
        with Timer("main-process") as t:
            try:
                # Inference
                audio = self.voiceChanger.inference(data)

                if hasattr(self, 'np_prev_audio1') == True:
                    np.set_printoptions(threshold=10000)
                    prev_overlap_start = -1 * crossfadeSize
                    prev_overlap = self.np_prev_audio1[prev_overlap_start:]
                    cur_overlap_start = -1 * (inputSize + crossfadeSize)
                    cur_overlap_end = -1 * inputSize
                    cur_overlap = audio[cur_overlap_start:cur_overlap_end]
                    print_convert_processing(
                        f" audio:{audio.shape}, prev_overlap:{prev_overlap.shape}, self.np_prev_strength:{self.np_prev_strength.shape}")
                    powered_prev = prev_overlap * self.np_prev_strength
                    print_convert_processing(
                        f" audio:{audio.shape}, cur_overlap:{cur_overlap.shape}, self.np_cur_strength:{self.np_cur_strength.shape}")
                    print_convert_processing(f" cur_overlap_strt:{cur_overlap_start}, cur_overlap_end{cur_overlap_end}")
                    powered_cur = cur_overlap * self.np_cur_strength
                    powered_result = powered_prev + powered_cur

                    cur = audio[-1 * inputSize:-1 * crossfadeSize]
                    result = np.concatenate([powered_result, cur], axis=0)
                    print_convert_processing(
                        f" overlap:{crossfadeSize}, current:{cur.shape[0]}, result:{result.shape[0]}... result should be same as input")
                    if cur.shape[0] != result.shape[0]:
                        print_convert_processing(f" current and result should be same as input")

                else:
                    result = np.zeros(4096).astype(np.int16)
                self.np_prev_audio1 = audio

            except Exception as e:
                print("VC PROCESSING!!!! EXCEPTION!!!", e)
                print(traceback.format_exc())
                if hasattr(self, "np_prev_audio1"):
                    del self.np_prev_audio1
                return np.zeros(1).astype(np.int16), [0, 0, 0]
        mainprocess_time = t.secs

        # 後処理
        with Timer("post-process") as t:
            result = result.astype(np.int16)
            if self.settings.inputSampleRate != processing_sampling_rate:
                outputData = resampy.resample(result, processing_sampling_rate, self.settings.inputSampleRate).astype(np.int16)
            else:
                outputData = result
            # outputData = result

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


##############
PRINT_CONVERT_PROCESSING = False
# PRINT_CONVERT_PROCESSING = True


def print_convert_processing(mess: str):
    if PRINT_CONVERT_PROCESSING == True:
        print(mess)


def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr


class Timer(object):
    def __init__(self, title: str):
        self.title = title

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
