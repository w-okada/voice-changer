from typing import Any, ClassVar, Optional, Union
from const import TMP_DIR, getModelType
import torch
import os
import traceback
import numpy as np
from dataclasses import dataclass, asdict
import resampy
from voice_changer.MultiprocessingWorker import WorkerManager


from voice_changer.IORecorder import IORecorder
from voice_changer.utils.VoiceChangerModel import AudioInOut


providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

STREAM_INPUT_FILE = os.path.join(TMP_DIR, "in.wav")
STREAM_OUTPUT_FILE = os.path.join(TMP_DIR, "out.wav")
STREAM_ANALYZE_FILE_DIO = os.path.join(TMP_DIR, "analyze-dio.png")
STREAM_ANALYZE_FILE_HARVEST = os.path.join(TMP_DIR, "analyze-harvest.png")


@dataclass
class VoiceChangerSettings():
    inputSampleRate: int = 24000  # 48000 or 24000
    recordIO: int = 0  # 0:off, 1:on

    # ↓mutableな物だけ列挙
    intData: ClassVar[list[str]] = ["inputSampleRate", "recordIO"]
    floatData: ClassVar[list[str]] = []
    strData: ClassVar[list[str]] = []


class VoiceChanger():

    def __init__(self, params):
        # 初期化
        self.wm = WorkerManager(getModelType(), params)
        self.settings = VoiceChangerSettings()

        gpu_num = torch.cuda.device_count()
        mps_enabled: bool = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()  # type: ignore

        print(f"VoiceChanger Initialized (GPU_NUM:{gpu_num}, mps_enabled:{mps_enabled})")

    def loadModel(
        self,
        config: str,
        pyTorch_model_file: Optional[str] = None,
        onnx_model_file: Optional[str] = None,
        clusterTorchModel: Optional[str] = None,
        feature_file: Optional[str] = None,
        index_file: Optional[str] = None,
        is_half: bool = True,
    ):
        self.wm.loadModel(config, pyTorch_model_file, onnx_model_file, clusterTorchModel, feature_file, index_file, is_half)

    def get_info(self):
        data = asdict(self.settings)
        return {**data, **self.wm.get_info()}

    def update_settings(self, key: str, val: Any):
        if key in self.settings.intData:
            if key == "recordIO":
                if hasattr(self, "ioRecorder"):
                    self.ioRecorder.close()
                if val == 1:
                    self.ioRecorder = IORecorder(STREAM_INPUT_FILE, STREAM_OUTPUT_FILE, self.settings.inputSampleRate)
            else:
                setattr(self.settings, key, int(val))
            return self.get_info()

        if key in self.settings.floatData:
            setattr(self.settings, key, float(val))
            return self.get_info()

        if key in self.settings.strData:
            setattr(self.settings, key, str(val))
            return self.get_info()

        self.wm.update_settings(key, val)

    #  receivedData: tuple of short
    def on_request(self, receivedData: AudioInOut) -> tuple[AudioInOut, list[Union[int, float]]]:
        self.wm.transmit_data(receivedData)
        outputData, perf = self.wm.output.get()


        if self.settings.recordIO == 1:
            self.ioRecorder.writeInput(receivedData)
            self.ioRecorder.writeOutput(outputData.tobytes())

        return outputData, [*perf]


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
