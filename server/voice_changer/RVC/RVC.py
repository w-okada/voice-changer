import sys
import os
import resampy
from voice_changer.RVC.ModelWrapper import ModelWrapper
from Exceptions import NoModeLoadedException

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
from const import HUBERT_ONNX_MODEL_PATH, TMP_DIR

import pyworld as pw

from voice_changer.RVC.custom_vc_infer_pipeline import VC
from infer_pack.models import SynthesizerTrnMs256NSFsid
from fairseq import checkpoint_utils
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class ModelSlot():
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    featureFile: str = ""
    indexFile: str = ""


@dataclass
class RVCSettings():
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "pm"  # pm or harvest
    tran: int = 20
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32
    clusterInferRatio: float = 0.1

    framework: str = "PyTorch"  # PyTorch or ONNX
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    configFile: str = ""
    modelSlots: list[ModelSlot] = field(
        default_factory=lambda: [
            ModelSlot(), ModelSlot(), ModelSlot()
        ]
    )
    indexRatio: float = 0
    rvcQuality: int = 0
    silenceFront: int = 1  # 0:off, 1:on
    modelSamplingRate: int = 48000
    modelSlotIndex: int = 0

    speakers: dict[str, int] = field(
        default_factory=lambda: {}
    )

    # ↓mutableな物だけ列挙
    intData = ["gpu", "dstId", "tran", "extraConvertSize", "rvcQuality", "modelSamplingRate", "silenceFront", "modelSlotIndex"]
    floatData = ["silentThreshold", "indexRatio"]
    strData = ["framework", "f0Detector"]


class RVC:
    def __init__(self, params):
        self.settings = RVCSettings()
        self.net_g = None
        self.onnx_session = None

        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params
        self.mps_enabled: bool = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        print("RVC initialization: ", params)
        print("mps: ", self.mps_enabled)

    def loadModel(self, props):

        # self.settings.pyTorchModelFile = props["files"]["pyTorchModelFilename"]
        # self.settings.onnxModelFile = props["files"]["onnxModelFilename"]

        # self.feature_file = props["files"]["featureFilename"]
        # self.index_file = props["files"]["indexFilename"]

        self.is_half = props["isHalf"]
        self.slot = props["slot"]

        self.settings.modelSlots[self.slot] = ModelSlot(
            pyTorchModelFile=props["files"]["pyTorchModelFilename"],
            onnxModelFile=props["files"]["onnxModelFilename"],
            featureFile=props["files"]["featureFilename"],
            indexFile=props["files"]["indexFilename"]
        )

        print("[Voice Changer] RVC loading... slot:", self.slot)

        try:
            hubert_path = self.params["hubert_base"]
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([hubert_path], suffix="",)
            model = models[0]
            model.eval()
            if self.is_half:
                model = model.half()
            self.hubert_model = model

        except Exception as e:
            print("EXCEPTION during loading hubert/contentvec model", e)

        self.switchModel(self.slot)

        return self.get_info()

    def switchModel(self, slot: int):
        print("[Voice Changer] Switch Model to:", slot)
        self.slot = slot
        pyTorchModelFile = self.settings.modelSlots[slot].pyTorchModelFile
        onnxModelFile = self.settings.modelSlots[slot].onnxModelFile
        # PyTorchモデル生成
        if pyTorchModelFile != None and pyTorchModelFile != "":
            cpt = torch.load(pyTorchModelFile, map_location="cpu")
            self.settings.modelSamplingRate = cpt["config"][-1]
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            net_g.eval()
            net_g.load_state_dict(cpt["weight"], strict=False)
            if self.is_half:
                net_g = net_g.half()
            self.net_g = net_g
        else:
            self.net_g = None

        # ONNXモデル生成
        if onnxModelFile != None and onnxModelFile != "":
            self.onnx_session = ModelWrapper(onnxModelFile)
        else:
            self.onnx_session = None

        self.feature_file = self.settings.modelSlots[slot].featureFile
        self.index_file = self.settings.modelSlots[slot].indexFile

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
            if key == "modelSlotIndex":
                self.switchModel(int(val))
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

    def generate_input(self, newData: any, inputSize: int, crossfadeSize: int, solaSearchFrame: int = 0):
        newData = newData.astype(np.float32) / 32768.0

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize

        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (128 - (convertSize % 128))

        self.audio_buffer = self.audio_buffer[-1 * convertSize:]  # 変換対象の部分だけ抽出

        crop = self.audio_buffer[-1 * (inputSize + crossfadeSize):-1 * (crossfadeSize)]  # 出力部分だけ切り出して音量を確認。(solaとの関係性について、現状は無考慮)
        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        return (self.audio_buffer, convertSize, vol)

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") == False or self.onnx_session == None:
            print("[Voice Changer] No onnx session.")
            raise NoModeLoadedException("ONNX")

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
            f0_method = self.settings.f0Detector
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
            raise NoModeLoadedException("pytorch")

        if self.settings.gpu < 0 or (self.gpu_num == 0 and self.mps_enabled == False):
            dev = torch.device("cpu")
        elif self.mps_enabled:
            dev = torch.device("mps")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        # print("device:", dev)

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
            f0_method = self.settings.f0Detector
            file_index = self.index_file if self.index_file != None else ""
            file_big_npy = self.feature_file if self.feature_file != None else ""
            index_rate = self.settings.indexRatio
            if_f0 = 1
            f0_file = None

            if self.settings.silenceFront == 0:
                audio_out = vc.pipeline(self.hubert_model, self.net_g, sid, audio, times, f0_up_key, f0_method,
                                        file_index, file_big_npy, index_rate, if_f0, f0_file=f0_file, silence_front=0)
            else:
                audio_out = vc.pipeline(self.hubert_model, self.net_g, sid, audio, times, f0_up_key, f0_method,
                                        file_index, file_big_npy, index_rate, if_f0, f0_file=f0_file, silence_front=self.settings.extraConvertSize / self.settings.modelSamplingRate)

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

    def export2onnx(self):
        if hasattr(self, "net_g") == False or self.net_g == None:
            print("[Voice Changer] export2onnx, No pyTorch session.")
            return {"status": "ng", "path": f""}
        if self.settings.pyTorchModelFile == None:
            print("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": f""}
        import voice_changer.RVC.export2onnx as onnxExporter

        output_file = os.path.splitext(os.path.basename(self.settings.pyTorchModelFile))[0] + ".onnx"
        output_file_simple = os.path.splitext(os.path.basename(self.settings.pyTorchModelFile))[0] + "_simple.onnx"
        output_path = os.path.join(TMP_DIR, output_file)
        output_path_simple = os.path.join(TMP_DIR, output_file_simple)

        if torch.cuda.device_count() > 0:
            onnxExporter.export2onnx(self.settings.pyTorchModelFile, output_path, output_path_simple, True)
        else:
            print("[Voice Changer] Warning!!! onnx export with float32. maybe size is doubled.")
            onnxExporter.export2onnx(self.settings.pyTorchModelFile, output_path, output_path_simple, False)

        return {"status": "ok", "path": f"/tmp/{output_file_simple}", "filename": output_file_simple}
