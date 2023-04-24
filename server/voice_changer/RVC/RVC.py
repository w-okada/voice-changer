import sys
import os
import json
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
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from .models import SynthesizerTrnMsNSFsid as SynthesizerTrnMsNSFsid_webui
from .models import SynthesizerTrnMsNSFsidNono as SynthesizerTrnMsNSFsidNono_webui

from .const import RVC_MODEL_TYPE_RVC, RVC_MODEL_TYPE_WEBUI
from fairseq import checkpoint_utils
providers = ['OpenVINOExecutionProvider', "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


@dataclass
class ModelSlot():
    pyTorchModelFile: str = ""
    onnxModelFile: str = ""
    featureFile: str = ""
    indexFile: str = ""
    defaultTrans: int = ""
    modelType: int = RVC_MODEL_TYPE_RVC
    samplingRate: int = -1
    f0: bool = True
    embChannels: int = 256
    samplingRateOnnx: int = -1
    f0Onnx: bool = True


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
        self.initialLoad = True
        self.settings = RVCSettings()

        self.inferenceing: bool = False

        self.net_g = None
        self.onnx_session = None
        self.feature_file = None
        self.index_file = None

        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params

        self.mps_enabled: bool = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        self.currentSlot = -1
        print("RVC initialization: ", params)
        print("mps: ", self.mps_enabled)

    def loadModel(self, props):
        self.is_half = props["isHalf"]
        self.tmp_slot = props["slot"]
        params_str = props["params"]
        params = json.loads(params_str)

        self.settings.modelSlots[self.tmp_slot] = ModelSlot(
            pyTorchModelFile=props["files"]["pyTorchModelFilename"],
            onnxModelFile=props["files"]["onnxModelFilename"],
            featureFile=props["files"]["featureFilename"],
            indexFile=props["files"]["indexFilename"],
            defaultTrans=params["trans"]
        )

        print("[Voice Changer] RVC loading... slot:", self.tmp_slot)

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

        if self.initialLoad:
            self.prepareModel(self.tmp_slot)
            self.settings.modelSlotIndex = self.tmp_slot
            self.currentSlot = self.settings.modelSlotIndex
            self.switchModel()
            self.initialLoad = False

        return self.get_info()

    def prepareModel(self, slot: int):
        print("[Voice Changer] Prepare Model of slot:", slot)
        pyTorchModelFile = self.settings.modelSlots[slot].pyTorchModelFile
        onnxModelFile = self.settings.modelSlots[slot].onnxModelFile
        # PyTorchモデル生成
        if pyTorchModelFile != None and pyTorchModelFile != "":
            print("[Voice Changer] Loading Pytorch Model...")
            cpt = torch.load(pyTorchModelFile, map_location="cpu")
            '''
            (1) オリジナルとrvc-webuiのモデル判定 ⇒ config全体の形状
            ■ ノーマル256
            [1025, 32, 192, 192, 768, 2, 6, 3, 0, '1', [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 48000]
            ■ ノーマル 768対応
            [1025, 32, 192, 192, 768, 2, 6, 3, 0, '1', [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 768, 48000]
            ⇒ 18: オリジナル, 19: rvc-webui

            (2-1) オリジナルのノーマルorPitchレス判定 ⇒ ckp["f0"]で判定
            0: ピッチレス, 1:ノーマル

            (2-2) rvc-webuiの、(256 or 768) x (ノーマルor pitchレス)判定 ⇒ 256, or 768 は17番目の要素で判定。, ノーマルor pitchレスはckp["f0"]で判定            
            '''

            # print("config shape:1::::", cpt["config"], cpt["f0"])
            # print("config shape:2::::", (cpt).keys)
            config_len = len(cpt["config"])
            if config_len == 18:
                self.settings.modelSlots[slot].modelType = RVC_MODEL_TYPE_RVC
                self.settings.modelSlots[slot].embChannels = 256
            else:
                self.settings.modelSlots[slot].modelType = RVC_MODEL_TYPE_WEBUI
                self.settings.modelSlots[slot].embChannels = cpt["config"][17]
            self.settings.modelSlots[slot].f0 = True if cpt["f0"] == 1 else False
            self.settings.modelSlots[slot].samplingRate = cpt["config"][-1]

            self.settings.modelSamplingRate = cpt["config"][-1]

            if self.settings.modelSlots[slot].modelType == RVC_MODEL_TYPE_RVC and self.settings.modelSlots[slot].f0 == True:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            elif self.settings.modelSlots[slot].modelType == RVC_MODEL_TYPE_RVC and self.settings.modelSlots[slot].f0 == False:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif self.settings.modelSlots[slot].modelType == RVC_MODEL_TYPE_WEBUI and self.settings.modelSlots[slot].f0 == True:
                net_g = SynthesizerTrnMsNSFsid_webui(**cpt["params"], is_half=self.is_half)
            elif self.settings.modelSlots[slot].modelType == RVC_MODEL_TYPE_WEBUI and self.settings.modelSlots[slot].f0 == False:
                ######################
                # TBD
                ######################
                print("webui non-f0 is not supported yet")
                net_g = SynthesizerTrnMsNSFsidNono_webui(**cpt["params"], is_half=self.is_half)

            else:
                print("unknwon")

            net_g.eval()
            net_g.load_state_dict(cpt["weight"], strict=False)
            if self.is_half:
                net_g = net_g.half()
            self.next_net_g = net_g
        else:
            print("[Voice Changer] Skip Loading Pytorch Model...")
            self.next_net_g = None

        # ONNXモデル生成
        if onnxModelFile != None and onnxModelFile != "":
            print("[Voice Changer] Loading ONNX Model...")
            self.next_onnx_session = ModelWrapper(onnxModelFile)
            self.settings.modelSlots[slot].samplingRateOnnx = self.next_onnx_session.getSamplingRate()
            self.settings.modelSlots[slot].f0Onnx = self.next_onnx_session.getF0()
            if self.settings.modelSlots[slot].samplingRate == -1:  # ONNXにsampling rateが入っていない
                self.settings.modelSlots[slot].samplingRate = self.settings.modelSamplingRate

            # ONNXがある場合は、ONNXの設定を優先
            self.settings.modelSlots[slot].samplingRate = self.settings.modelSlots[slot].samplingRateOnnx
            self.settings.modelSlots[slot].f0 = self.settings.modelSlots[slot].f0Onnx

        else:
            print("[Voice Changer] Skip Loading ONNX Model...")
            self.next_onnx_session = None

        self.next_feature_file = self.settings.modelSlots[slot].featureFile
        self.next_index_file = self.settings.modelSlots[slot].indexFile
        self.next_trans = self.settings.modelSlots[slot].defaultTrans
        print("[Voice Changer] Prepare done.",)
        return self.get_info()

    def switchModel(self):
        print("[Voice Changer] Switching model..",)
        # del self.net_g
        # del self.onnx_session
        self.net_g = self.next_net_g
        self.onnx_session = self.next_onnx_session
        self.feature_file = self.next_feature_file
        self.index_file = self.next_index_file
        self.settings.tran = self.next_trans
        self.next_net_g = None
        self.next_onnx_session = None
        print("[Voice Changer] Switching model..done",)

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
            if key == "gpu" and val >= 0 and val < self.gpu_num and self.onnx_session != None:
                providers = self.onnx_session.get_providers()
                print("Providers:", providers)
                if "CUDAExecutionProvider" in providers:
                    provider_options = [{'device_id': self.settings.gpu}]
                    self.onnx_session.set_providers(providers=["CUDAExecutionProvider"], provider_options=provider_options)
            if key == "modelSlotIndex":
                # self.switchModel(int(val))
                self.tmp_slot = int(val)
                self.prepareModel(self.tmp_slot)
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

            f0 = self.settings.modelSlots[self.currentSlot].f0
            embChannels = self.settings.modelSlots[self.currentSlot].embChannels
            audio_out = vc.pipeline(self.hubert_model, self.onnx_session, sid, audio, times, f0_up_key, f0_method,
                                    file_index, file_big_npy, index_rate, if_f0, f0_file=f0_file, silence_front=self.settings.extraConvertSize / self.settings.modelSamplingRate, f0=f0, embChannels=embChannels)
            result = audio_out * np.sqrt(vol)

        return result

    def _pyTorch_inference(self, data):
        if hasattr(self, "net_g") == False or self.net_g == None:
            print("[Voice Changer] No pyTorch session.", hasattr(self, "net_g"), self.net_g)
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

            f0 = self.settings.modelSlots[self.currentSlot].f0

            embChannels = self.settings.modelSlots[self.currentSlot].embChannels
            audio_out = vc.pipeline(self.hubert_model, self.net_g, sid, audio, times, f0_up_key, f0_method,
                                    file_index, file_big_npy, index_rate, if_f0, f0_file=f0_file, silence_front=self.settings.extraConvertSize / self.settings.modelSamplingRate, f0=f0, embChannels=embChannels)

            result = audio_out * np.sqrt(vol)

        return result

    def inference(self, data):
        # if self.settings.modelSlotIndex < -1:
        #     print("[Voice Changer] No model uploaded.")
        #     raise NoModeLoadedException("model_common")

        if self.currentSlot != self.settings.modelSlotIndex:
            print(f"Switch model {self.currentSlot} -> {self.settings.modelSlotIndex}")
            self.currentSlot = self.settings.modelSlotIndex
            self.switchModel()

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

        pyTorchModelFile = self.settings.modelSlots[self.settings.modelSlotIndex].pyTorchModelFile  # inference前にexportできるようにcurrentSlotではなくslot

        if pyTorchModelFile == None:
            print("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": f""}
        import voice_changer.RVC.export2onnx as onnxExporter

        output_file = os.path.splitext(os.path.basename(pyTorchModelFile))[0] + ".onnx"
        output_file_simple = os.path.splitext(os.path.basename(pyTorchModelFile))[0] + "_simple.onnx"
        output_path = os.path.join(TMP_DIR, output_file)
        output_path_simple = os.path.join(TMP_DIR, output_file_simple)
        metadata = {
            "application": "VC_CLIENT",
            "version": "1",
            "ModelType": self.settings.modelSlots[self.settings.modelSlotIndex].modelType,
            "samplingRate": self.settings.modelSlots[self.settings.modelSlotIndex].samplingRate,
            "f0": self.settings.modelSlots[self.settings.modelSlotIndex].f0,
            "embChannels": self.settings.modelSlots[self.settings.modelSlotIndex].embChannels,
        }

        if torch.cuda.device_count() > 0:
            onnxExporter.export2onnx(pyTorchModelFile, output_path, output_path_simple, True, metadata)
        else:
            print("[Voice Changer] Warning!!! onnx export with float32. maybe size is doubled.")
            onnxExporter.export2onnx(pyTorchModelFile, output_path, output_path_simple, False, metadata)

        return {"status": "ok", "path": f"/tmp/{output_file_simple}", "filename": output_file_simple}
