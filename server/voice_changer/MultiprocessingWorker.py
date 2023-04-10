import time, traceback, numpy as np, resampy

from typing import Any, ClassVar, Literal, Optional, TypeAlias, cast
from const import TMP_DIR
from dataclasses import dataclass, asdict
from multiprocessing import Process, Queue, Value
from multiprocessing.sharedctypes import Synchronized
from ctypes import c_bool
from const import ModelType


from voice_changer.utils.VoiceChangerModel import VoiceChangerModel, AudioInOut
from voice_changer.utils.Timer import Timer


WorkerCommand: TypeAlias = Literal['load_model', 'set_options', 'get_info']

##############
PRINT_CONVERT_PROCESSING: bool = False
# PRINT_CONVERT_PROCESSING = True


def print_convert_processing(mess: str):
    if PRINT_CONVERT_PROCESSING == True:
        print(mess)


@dataclass
class WorkerSettings():
    inputSampleRate: int = 24000  # 48000 or 24000

    crossFadeOffsetRate: float = 0.1
    crossFadeEndRate: float = 0.9
    crossFadeOverlapSize: int = 4096

    # ↓mutableな物だけ列挙
    intData: ClassVar[list[str]] = ["inputSampleRate", "crossFadeOverlapSize", "recordIO"]
    floatData: ClassVar[list[str]] = ["crossFadeOffsetRate", "crossFadeEndRate"]
    strData: ClassVar[list[str]] = []


class Worker:
    def __init__(
        self,
        model: ModelType,
        params: Any,
        running: 'Synchronized[c_bool]',
        data: 'Queue[AudioInOut]',
        cmd_send: 'Queue[tuple[WorkerCommand, dict[str, Any]]]',
        cmd_recv: 'Queue[Any]',
        output: 'Queue[tuple[AudioInOut, tuple[float, float, float]]]',
    ) -> None:
        self.settings = WorkerSettings()
        self.running = running
        self.data = data
        self.command_channel = cmd_send
        self.command_recv = cmd_recv
        self.output = output
        self.mty = model
        self.currentCrossFadeOffsetRate = 0
        self.currentCrossFadeEndRate = 0
        self.currentCrossFadeOverlapSize = 0
        self.crossfadeSize = 0
        self.voiceChanger = self.get_models_by_type_name(model, params)

    def get_models_by_type_name(self, mty: ModelType, params: Any) -> VoiceChangerModel:
        if mty == "MMVCv15":
            from voice_changer.MMVCv15.MMVCv15 import MMVCv15
            return MMVCv15()  # type: ignore
        elif mty == "MMVCv13":
            from voice_changer.MMVCv13.MMVCv13 import MMVCv13
            return MMVCv13()
        elif mty == "so-vits-svc-40v2":
            from voice_changer.SoVitsSvc40v2.SoVitsSvc40v2 import SoVitsSvc40v2
            return SoVitsSvc40v2(params)
        elif mty == "so-vits-svc-40" or mty == "so-vits-svc-40_c":
            from voice_changer.SoVitsSvc40.SoVitsSvc40 import SoVitsSvc40
            return SoVitsSvc40(params)
        elif mty == "DDSP-SVC":
            from voice_changer.DDSP_SVC.DDSP_SVC import DDSP_SVC
            return DDSP_SVC(params)
        elif mty == "RVC":
            from voice_changer.RVC.RVC import RVC
            return RVC(params)
        else:
            from voice_changer.MMVCv13.MMVCv13 import MMVCv13
            return MMVCv13()

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
        if self.mty == "MMVCv15" or self.mty == "MMVCv13":
            return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file)
        elif self.mty == "so-vits-svc-40" or self.mty == "so-vits-svc-40_c" or self.mty == "so-vits-svc-40v2":
            return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file, clusterTorchModel)
        elif self.mty == "RVC":
            return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file, feature_file, index_file, is_half)
        else:
            return self.voiceChanger.loadModel(config, pyTorch_model_file, onnx_model_file, clusterTorchModel)

    def get_info(self):
        data = asdict(self.settings)
        data.update(self.voiceChanger.get_info())
        return data

    def update_settings(self, key: str, value: Any):
        if key in self.settings.intData:
            setattr(self.settings, key, int(value))
            if key == "crossFadeOffsetRate" or key == "crossFadeEndRate":
                self.crossfadeSize = 0
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(value))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(value))
        else:
            ret = self.voiceChanger.update_settings(key, value)
            if not ret:
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
            if hasattr(self, 'np_prev_audio1'):
                delattr(self, "np_prev_audio1")

    def infer_output(self, receivedData: AudioInOut):
        processing_sampling_rate = self.voiceChanger.get_processing_sampling_rate()

        print_convert_processing(f"------------ Convert processing.... ------------")
        # 前処理
        with Timer("pre-process") as t:
            if self.settings.inputSampleRate != processing_sampling_rate:
                newData = cast(AudioInOut, resampy.resample(receivedData, self.settings.inputSampleRate, processing_sampling_rate))
            else:
                newData = receivedData
            inputSize = newData.shape[0]
            crossfadeSize = min(self.settings.crossFadeOverlapSize, inputSize)

            print_convert_processing(
                f" Input data size: {receivedData.shape[0]}/{self.settings.inputSampleRate}hz {inputSize}/{processing_sampling_rate}hz")
            print_convert_processing(
                f" Crossfade data size: crossfade:{crossfadeSize}, crossfade setting:{self.settings.crossFadeOverlapSize}, input size:{inputSize}")

            print_convert_processing(f" Convert data size of {inputSize + crossfadeSize} (+ extra size)")
            print_convert_processing(f"         will be cropped:{-1 * (inputSize + crossfadeSize)}, {-1 * (crossfadeSize)}")

            self._generate_strength(crossfadeSize)
            data = self.voiceChanger.generate_input(newData, inputSize, crossfadeSize)

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
                return np.zeros(1).astype(np.int16), (0., 0., 0.)
        mainprocess_time = t.secs

        # 後処理
        with Timer("post-process") as t:
            result = result.astype(np.int16)
            if self.settings.inputSampleRate != processing_sampling_rate:
                outputData = cast(AudioInOut, resampy.resample(result, processing_sampling_rate, self.settings.inputSampleRate).astype(np.int16))
            else:
                outputData = result
            # outputData = result

            print_convert_processing(
                f" Output data size of {result.shape[0]}/{processing_sampling_rate}hz {outputData.shape[0]}/{self.settings.inputSampleRate}hz")

        postprocess_time = t.secs

        print_convert_processing(f" [fin] Input/Output size:{receivedData.shape[0]},{outputData.shape[0]}")
        return outputData, (preprocess_time, mainprocess_time, postprocess_time)

    def dispatch_command(self, name: WorkerCommand, args: dict[str, Any]):
        if name == "load_model":
            return self.loadModel(**args)
        elif name == "set_options":
            return self.update_settings(**args)
        elif name == "get_info":
            return self.get_info()

    def loop(self):
        while self.running.value:
            if not self.command_channel.empty():
                name, args = self.command_channel.get()
                try:
                    self.command_recv.put(self.dispatch_command(name, args))
                except Exception as e:
                    print(f"ERROR: {e}")
                    self.command_recv.put({"error": str(e)})
            if self.data.empty():
                time.sleep(0)
                continue
            out = self.infer_output(self.data.get())
            self.output.put(out)


def _run_worker(
    model: ModelType,
    params: Any,
    running: 'Synchronized[c_bool]',
    data: 'Queue[AudioInOut]',
    cmd_send: 'Queue[tuple[WorkerCommand, dict[str, Any]]]',
    cmd_recv: 'Queue[Any]',
    output: 'Queue[tuple[AudioInOut, tuple[float, float, float]]]',
):
    Worker(
        model,
        params,
        running,
        data,
        cmd_send,
        cmd_recv,
        output,
    ).loop()


class WorkerManager:
    def __init__(self, model: ModelType, params: Any) -> None:
        self.running = cast('Synchronized[c_bool]', Value(c_bool, c_bool(True)))
        self.command_channel: 'Queue[tuple[WorkerCommand, dict[str, Any]]]' = Queue()
        self.command_result: 'Queue[Any]' = Queue()
        self.data: 'Queue[AudioInOut]' = Queue()
        self.output: 'Queue[tuple[AudioInOut, tuple[float, float, float]]]' = Queue()
        self.process = Process(target=_run_worker, args=(model, params, self.running, self.data, self.command_channel, self.command_result, self.output))

    def start(self):
        self.process.start()

    def stop(self):
        self.running.value = c_bool(False)
        self.process.join()

    def _send_command(self, name: WorkerCommand, **args: Any):
        self.command_channel.put((name, args))
        return self.command_result.get()

    def loadModel(
        self,
        config: str,
        pyTorch_model_file: Optional[str] = None,
        onnx_model_file: Optional[str] = None,
        clusterTorchModel: Optional[str] = None,
        feature_file: Optional[str] = None,
        index_file: Optional[str] = None,
        is_half: bool = True,
    ) -> dict[str, Any]:
        return self._send_command(
            "load_model",
            config=config,
            pyTorch_model_file=pyTorch_model_file,
            onnx_model_file=onnx_model_file,
            clusterTorchModel=clusterTorchModel,
            feature_file=feature_file,
            index_file=index_file,
            is_half=is_half
        )

    def update_settings(self, key: str, value: Any) -> dict[str, Any]:
        return self._send_command("set_options", key=key, value=value)
    
    def get_info(self) -> dict[str, Any]:
        return self._send_command("get_info")

    def transmit_data(self, data: AudioInOut):
        self.data.put(data)

    def receive_output(self) -> tuple[AudioInOut, tuple[float, float, float]]:
        if self.output.empty():
            return np.zeros(1).astype(np.int16), (0., 0., 0.)
        return self.output.get()
