import sys
import os
import resampy
from voice_changer.RVC.ModelWrapper import ModelWrapper

# avoiding parse arg error in RVC
sys.argv = ["MMVCServerSIO.py"]

if sys.platform.startswith('darwin'):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "RVC")
    sys.path.append(modulePath)
else:
    sys.path.append("RVC")

import io
from dataclasses import dataclass, asdict, field
from functools import reduce
import numpy as np
import torch
import onnxruntime
# onnxruntime.set_default_logger_severity(3)
from const import HUBERT_ONNX_MODEL_PATH

import pyworld as pw

from voice_changer.RVC.custom_vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid
from fairseq import checkpoint_utils
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class RVCSettings():
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "dio"  # dio or harvest
    tran: int = 20
    noiceScale: float = 0.3
    predictF0: int = 0  # 0:False, 1:True
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32
    clusterInferRatio: float = 0.1

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""

    indexRatio: float = 0
    rvcQuality: int = 0
    modelSamplingRate: int = 48000

    speakers: dict[str, int] = field(
        default_factory=lambda: {}
    )

    # ↓mutableな物だけ列挙
    intData = ["gpu", "dstId", "tran", "predictF0", "extraConvertSize", "rvcQuality", "modelSamplingRate"]
    floatData = ["noiceScale", "silentThreshold", "indexRatio"]
    strData = ["framework", "f0Detector"]


class RVC:
    def __init__(self, params):
        self.settings = RVCSettings()
        self.net_g = None
        self.onnx_session = None

        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params
        print("RVC initialization: ", params)

    def loadModel(self, config: str, pyTorch_model_file: str = None, onnx_model_file: str = None, feature_file: str = None, index_file: str = None, is_half: bool = True):
        self.settings.configFile = config
        self.feature_file = feature_file
        self.index_file = index_file
        self.is_half = is_half

        try:
            hubert_path = self.params["hubert"]
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([hubert_path], suffix="",)
            model = models[0]
            model.eval()
            if self.is_half:
                model = model.half()
            self.hubert_model = model

        except Exception as e:
            print("EXCEPTION during loading hubert/contentvec model", e)

        if pyTorch_model_file != None:
            self.settings.pyTorchModelFile = pyTorch_model_file
        if onnx_model_file:
            self.settings.onnxModelFile = onnx_model_file

        # PyTorchモデル生成
        if pyTorch_model_file != None:
            cpt = torch.load(pyTorch_model_file, map_location="cpu")
            self.settings.modelSamplingRate = cpt["config"][-1]
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            net_g.eval()
            net_g.load_state_dict(cpt["weight"], strict=False)
            if self.is_half:
                net_g = net_g.half()
            self.net_g = net_g

        # ONNXモデル生成
        if onnx_model_file != None:
            self.onnx_session = ModelWrapper(onnx_model_file, is_half=self.is_half)
        return self.get_info()

    def update_settings(self, key: str, val: any):
        if key == "onnxExecutionProvider" and self.onnx_session != None:
            if val == "CUDAExecutionProvider":
                if self.settings.gpu < 0 or self.settings.gpu >= self.gpu_num:
                    self.settings.gpu = 0
                provider_options = [{'device_id': self.settings.gpu}]
                self.onnx_session.set_providers(providers=[val], provider_options=provider_options)
                if hasattr(self, "hubert_onnx"):
                    self.hubert_onnx.set_providers(providers=[val], provider_options=provider_options)
            else:
                self.onnx_session.set_providers(providers=[val])
                if hasattr(self, "hubert_onnx"):
                    self.hubert_onnx.set_providers(providers=[val])
        elif key == "onnxExecutionProvider" and self.onnx_session == None:
            print("Onnx is not enabled. Please load model.")
            return False
        elif key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "gpu" and val >= 0 and val < self.gpu_num and self.onnx_session != None:
                providers = self.onnx_session.get_providers()
                print("Providers:", providers)
                if "CUDAExecutionProvider" in providers:
                    provider_options = [{'device_id': self.settings.gpu}]
                    self.onnx_session.set_providers(providers=["CUDAExecutionProvider"], provider_options=provider_options)
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            return False

        return True

    def get_info(self):
        data = asdict(self.settings)

        data["onnxExecutionProviders"] = self.onnx_session.get_providers() if self.onnx_session != None else []
        files = ["configFile", "pyTorchModelFile", "onnxModelFile"]
        for f in files:
            if data[f] != None and os.path.exists(data[f]):
                data[f] = os.path.basename(data[f])
            else:
                data[f] = ""

        return data

    def get_processing_sampling_rate(self):
        return self.settings.modelSamplingRate

    def generate_input(self, newData: any, inputSize: int, crossfadeSize: int):
        newData = newData.astype(np.float32) / 32768.0

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + self.settings.extraConvertSize

        # if convertSize % self.hps.data.hop_length != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            # convertSize = convertSize + (self.hps.data.hop_length - (convertSize % self.hps.data.hop_length))
            convertSize = convertSize + (128 - (convertSize % 128))

        self.audio_buffer = self.audio_buffer[-1 * convertSize:]  # 変換対象の部分だけ抽出

        crop = self.audio_buffer[-1 * (inputSize + crossfadeSize):-1 * (crossfadeSize)]
        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        return (self.audio_buffer, convertSize, vol)

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") == False or self.onnx_session == None:
            print("[Voice Changer] No onnx session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        self.hubert_model = self.hubert_model.to(dev)

        audio = data[0]
        convertSize = data[1]
        vol = data[2]

        audio = resampy.resample(audio, self.settings.modelSamplingRate, 16000)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        with torch.no_grad():
            repeat = 3 if self.is_half else 1
            repeat *= self.settings.rvcQuality  # 0 or 3
            vc = VC(self.settings.modelSamplingRate, dev, self.is_half, repeat)
            sid = 0
            times = [0, 0, 0]
            f0_up_key = self.settings.tran
            f0_method = "pm" if self.settings.f0Detector == "dio" else "harvest"
            file_index = self.index_file if self.index_file != None else ""
            file_big_npy = self.feature_file if self.feature_file != None else ""
            index_rate = self.settings.indexRatio
            if_f0 = 1
            f0_file = None

            audio_out = vc.pipeline(self.hubert_model, self.onnx_session, sid, audio, times, f0_up_key, f0_method,
                                    file_index, file_big_npy, index_rate, if_f0, f0_file=f0_file)
            result = audio_out * np.sqrt(vol)

        return result

    def _pyTorch_inference(self, data):
        if hasattr(self, "net_g") == False or self.net_g == None:
            print("[Voice Changer] No pyTorch session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        self.hubert_model = self.hubert_model.to(dev)
        self.net_g = self.net_g.to(dev)

        audio = data[0]
        convertSize = data[1]
        vol = data[2]
        audio = resampy.resample(audio, self.settings.modelSamplingRate, 16000)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        with torch.no_grad():
            repeat = 3 if self.is_half else 1
            repeat *= self.settings.rvcQuality  # 0 or 3
            vc = VC(self.settings.modelSamplingRate, dev, self.is_half, repeat)
            sid = 0
            times = [0, 0, 0]
            f0_up_key = self.settings.tran
            f0_method = "pm" if self.settings.f0Detector == "dio" else "harvest"
            file_index = self.index_file if self.index_file != None else ""
            file_big_npy = self.feature_file if self.feature_file != None else ""
            index_rate = self.settings.indexRatio
            if_f0 = 1
            f0_file = None

            audio_out = vc.pipeline(self.hubert_model, self.net_g, sid, audio, times, f0_up_key, f0_method,
                                    file_index, file_big_npy, index_rate, if_f0, f0_file=f0_file)
            result = audio_out * np.sqrt(vol)

        return result

    def inference(self, data):
        if self.settings.framework == "ONNX":
            audio = self._onnx_inference(data)
        else:
            audio = self._pyTorch_inference(data)
        return audio

    def __del__(self):
        del self.net_g
        del self.onnx_session

        remove_path = os.path.join("RVC")
        sys.path = [x for x in sys.path if x.endswith(remove_path) == False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("RVC" + os.path.sep) >= 0:
                    print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception as e:
                pass
