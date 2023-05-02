import sys
import os
import json
import resampy
from voice_changer.RVC.MergeModel import merge_model
from voice_changer.RVC.MergeModelRequest import MergeModelRequest
from voice_changer.RVC.ModelWrapper import ModelWrapper
from Exceptions import NoModeLoadedException
from voice_changer.RVC.RVCSettings import RVCSettings
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.utils.LoadModelParams import FilePaths, LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

from dataclasses import asdict
from typing import cast
import numpy as np
import torch

# from fairseq import checkpoint_utils
import traceback
import faiss

from const import TMP_DIR, UPLOAD_DIR  # type:ignore


# avoiding parse arg error in RVC
sys.argv = ["MMVCServerSIO.py"]

if sys.platform.startswith("darwin"):
    baseDir = [x for x in sys.path if x.endswith("Contents/MacOS")]
    if len(baseDir) != 1:
        print("baseDir should be only one ", baseDir)
        sys.exit()
    modulePath = os.path.join(baseDir[0], "RVC")
    sys.path.append(modulePath)
else:
    sys.path.append("RVC")


from .models import SynthesizerTrnMsNSFsid as SynthesizerTrnMsNSFsid_webui
from .models import SynthesizerTrnMsNSFsidNono as SynthesizerTrnMsNSFsidNono_webui
from .const import RVC_MODEL_TYPE_RVC, RVC_MODEL_TYPE_WEBUI
from voice_changer.RVC.custom_vc_infer_pipeline import VC
from infer_pack.models import (  # type:ignore
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
)

providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


class RVC:
    audio_buffer: AudioInOut | None = None
    embedder: Embedder | None = None

    def __init__(self, params: VoiceChangerParams):
        self.initialLoad = True
        self.settings = RVCSettings()

        self.net_g = None
        self.onnx_session = None
        self.feature_file = None
        self.index_file = None

        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params

        self.mps_enabled: bool = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        self.currentSlot = -1
        print("RVC initialization: ", params)
        print("mps: ", self.mps_enabled)

    def loadModel(self, props: LoadModelParams):
        """
        loadModelはスロットへのエントリ（推論向けにはロードしない）。
        例外的に、まだ一つも推論向けにロードされていない場合は、ロードする。
        """
        self.is_half = props.isHalf
        tmp_slot = props.slot
        params_str = props.params
        params = json.loads(params_str)

        self.settings.modelSlots[
            tmp_slot
        ].pyTorchModelFile = props.files.pyTorchModelFilename
        self.settings.modelSlots[tmp_slot].onnxModelFile = props.files.onnxModelFilename
        self.settings.modelSlots[tmp_slot].featureFile = props.files.featureFilename
        self.settings.modelSlots[tmp_slot].indexFile = props.files.indexFilename
        self.settings.modelSlots[tmp_slot].defaultTrans = (
            params["trans"] if "trans" in params else 0
        )

        isONNX = (
            True
            if self.settings.modelSlots[tmp_slot].onnxModelFile is not None
            else False
        )

        # メタデータ設定
        if isONNX:
            self._setInfoByONNX(
                tmp_slot, self.settings.modelSlots[tmp_slot].onnxModelFile
            )
        else:
            self._setInfoByPytorch(
                tmp_slot, self.settings.modelSlots[tmp_slot].pyTorchModelFile
            )

        print(
            f"[Voice Changer] RVC loading... slot:{tmp_slot}",
            asdict(self.settings.modelSlots[tmp_slot]),
        )
        # hubertロード
        # try:
        #     hubert_path = self.params.hubert_base
        #     hubert_path_jp = self.params.hubert_base_jp
        #     print(hubert_path, hubert_path_jp)

        #     models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        #         [hubert_path],
        #         suffix="",
        #     )
        #     model = models[0]
        #     model.eval()
        #     if self.is_half:
        #         model = model.half()
        #     self.hubert_model = model

        # except Exception as e:
        #     print("EXCEPTION during loading hubert/contentvec model", e)
        #     print("          hubert_path:", hubert_path)

        # 初回のみロード
        if self.initialLoad or tmp_slot == self.currentSlot:
            self.prepareModel(tmp_slot)
            self.settings.modelSlotIndex = tmp_slot
            self.currentSlot = self.settings.modelSlotIndex
            self.switchModel()
            self.initialLoad = False

        return self.get_info()

    def _setInfoByPytorch(self, slot, file):
        cpt = torch.load(file, map_location="cpu")
        config_len = len(cpt["config"])
        if config_len == 18:
            self.settings.modelSlots[slot].modelType = RVC_MODEL_TYPE_RVC
            self.settings.modelSlots[slot].embChannels = 256
            self.settings.modelSlots[slot].embedder = "hubert_base"
        else:
            self.settings.modelSlots[slot].modelType = RVC_MODEL_TYPE_WEBUI
            self.settings.modelSlots[slot].embChannels = cpt["config"][17]
            self.settings.modelSlots[slot].embedder = cpt["embedder_name"]
            if self.settings.modelSlots[slot].embedder.endswith("768"):
                self.settings.modelSlots[slot].embedder = self.settings.modelSlots[
                    slot
                ].embedder[:-3]

        self.settings.modelSlots[slot].f0 = True if cpt["f0"] == 1 else False
        self.settings.modelSlots[slot].samplingRate = cpt["config"][-1]

        # self.settings.modelSamplingRate = cpt["config"][-1]

    def _setInfoByONNX(self, slot, file):
        tmp_onnx_session = ModelWrapper(file)
        self.settings.modelSlots[slot].modelType = tmp_onnx_session.getModelType()
        self.settings.modelSlots[slot].embChannels = tmp_onnx_session.getEmbChannels()
        self.settings.modelSlots[slot].embedder = tmp_onnx_session.getEmbedder()
        self.settings.modelSlots[slot].f0 = tmp_onnx_session.getF0()
        self.settings.modelSlots[slot].samplingRate = tmp_onnx_session.getSamplingRate()
        self.settings.modelSlots[slot].deprecated = tmp_onnx_session.getDeprecated()

    def prepareModel(self, slot: int):
        if slot < 0:
            return self.get_info()
        print("[Voice Changer] Prepare Model of slot:", slot)
        onnxModelFile = self.settings.modelSlots[slot].onnxModelFile
        isONNX = (
            True if self.settings.modelSlots[slot].onnxModelFile is not None else False
        )

        # モデルのロード
        if isONNX:
            print("[Voice Changer] Loading ONNX Model...")
            self.next_onnx_session = ModelWrapper(onnxModelFile)
            self.next_net_g = None
        else:
            print("[Voice Changer] Loading Pytorch Model...")
            torchModelSlot = self.settings.modelSlots[slot]
            cpt = torch.load(torchModelSlot.pyTorchModelFile, map_location="cpu")

            if (
                torchModelSlot.modelType == RVC_MODEL_TYPE_RVC
                and torchModelSlot.f0 is True
            ):
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            elif (
                torchModelSlot.modelType == RVC_MODEL_TYPE_RVC
                and torchModelSlot.f0 is False
            ):
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif (
                torchModelSlot.modelType == RVC_MODEL_TYPE_WEBUI
                and torchModelSlot.f0 is True
            ):
                net_g = SynthesizerTrnMsNSFsid_webui(
                    **cpt["params"], is_half=self.is_half
                )
            else:
                net_g = SynthesizerTrnMsNSFsidNono_webui(
                    **cpt["params"], is_half=self.is_half
                )
            net_g.eval()
            net_g.load_state_dict(cpt["weight"], strict=False)

            if self.is_half:
                net_g = net_g.half()

            self.next_net_g = net_g
            self.next_onnx_session = None

        # Indexのロード
        print("[Voice Changer] Loading index...")
        self.next_feature_file = self.settings.modelSlots[slot].featureFile
        self.next_index_file = self.settings.modelSlots[slot].indexFile

        if (
            self.settings.modelSlots[slot].featureFile is not None
            and self.settings.modelSlots[slot].indexFile is not None
        ):
            if (
                os.path.exists(self.settings.modelSlots[slot].featureFile) is True
                and os.path.exists(self.settings.modelSlots[slot].indexFile) is True
            ):
                try:
                    self.next_index = faiss.read_index(
                        self.settings.modelSlots[slot].indexFile
                    )
                    self.next_feature = np.load(
                        self.settings.modelSlots[slot].featureFile
                    )
                except:
                    print("[Voice Changer] load index failed. Use no index.")
                    traceback.print_exc()
                    self.next_index = self.next_feature = None
            else:
                print("[Voice Changer] Index file is not found. Use no index.")
                self.next_index = self.next_feature = None
        else:
            self.next_index = self.next_feature = None

        self.next_trans = self.settings.modelSlots[slot].defaultTrans
        self.next_samplingRate = self.settings.modelSlots[slot].samplingRate
        self.next_embedder = self.settings.modelSlots[slot].embedder
        self.next_framework = (
            "ONNX" if self.next_onnx_session is not None else "PyTorch"
        )
        print("[Voice Changer] Prepare done.")
        return self.get_info()

    def switchModel(self):
        print("[Voice Changer] Switching model..")
        # del self.net_g
        # del self.onnx_session
        try:
            self.embedder = EmbedderManager.getEmbedder(
                self.next_embedder,
                self.params.hubert_base,
                True,
                torch.device("cuda:0"),
            )
        except Exception as e:
            print("[Voice Changer] load hubert error", e)
            traceback.print_exc()

        self.net_g = self.next_net_g
        self.onnx_session = self.next_onnx_session
        self.feature_file = self.next_feature_file
        self.index_file = self.next_index_file
        self.feature = self.next_feature
        self.index = self.next_index
        self.settings.tran = self.next_trans
        self.settings.framework = self.next_framework
        self.settings.modelSamplingRate = self.next_samplingRate
        self.next_net_g = None
        self.next_onnx_session = None
        print(
            "[Voice Changer] Switching model..done",
        )

    def update_settings(self, key: str, val: int | float | str):
        if key == "onnxExecutionProvider" and self.onnx_session is not None:
            if val == "CUDAExecutionProvider":
                if self.settings.gpu < 0 or self.settings.gpu >= self.gpu_num:
                    self.settings.gpu = 0
                provider_options = [{"device_id": self.settings.gpu}]
                self.onnx_session.set_providers(
                    providers=[val], provider_options=provider_options
                )
                if hasattr(self, "hubert_onnx"):
                    self.hubert_onnx.set_providers(
                        providers=[val], provider_options=provider_options
                    )
            else:
                self.onnx_session.set_providers(providers=[val])
                if hasattr(self, "hubert_onnx"):
                    self.hubert_onnx.set_providers(providers=[val])
        elif key == "onnxExecutionProvider" and self.onnx_session is None:
            print("Onnx is not enabled. Please load model.")
            return False
        elif key in self.settings.intData:
            val = cast(int, val)
            if (
                key == "gpu"
                and val >= 0
                and val < self.gpu_num
                and self.onnx_session is not None
            ):
                providers = self.onnx_session.get_providers()
                print("Providers:", providers)
                if "CUDAExecutionProvider" in providers:
                    provider_options = [{"device_id": self.settings.gpu}]
                    self.onnx_session.set_providers(
                        providers=["CUDAExecutionProvider"],
                        provider_options=provider_options,
                    )
            if key == "modelSlotIndex":
                if int(val) < 0:
                    return True
                # self.switchModel(int(val))
                val = int(val) % 1000  # Quick hack for same slot is selected
                self.prepareModel(val)
                self.currentSlot = -1
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

        data["onnxExecutionProviders"] = (
            self.onnx_session.get_providers() if self.onnx_session is not None else []
        )
        files = ["configFile", "pyTorchModelFile", "onnxModelFile"]
        for f in files:
            if data[f] is not None and os.path.exists(data[f]):
                data[f] = os.path.basename(data[f])
            else:
                data[f] = ""

        return data

    def get_processing_sampling_rate(self):
        return self.settings.modelSamplingRate

    def generate_input(
        self,
        newData: AudioInOut,
        inputSize: int,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / 32768.0

        if self.audio_buffer is not None:
            # 過去のデータに連結
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)
        else:
            self.audio_buffer = newData

        convertSize = (
            inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize
        )

        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (128 - (convertSize % 128))

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        # 出力部分だけ切り出して音量を確認。(TODO:段階的消音にする)
        cropOffset = -1 * (inputSize + crossfadeSize)
        cropEnd = -1 * (crossfadeSize)
        crop = self.audio_buffer[cropOffset:cropEnd]
        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        return (self.audio_buffer, convertSize, vol)

    def _onnx_inference(self, data):
        if hasattr(self, "onnx_session") is False or self.onnx_session is None:
            print("[Voice Changer] No onnx session.")
            raise NoModeLoadedException("ONNX")

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        # self.hubert_model = self.hubert_model.to(dev)
        self.embedder = self.embedder.to(dev)

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
            f0_up_key = self.settings.tran
            f0_method = self.settings.f0Detector
            index_rate = self.settings.indexRatio
            if_f0 = 1 if self.settings.modelSlots[self.currentSlot].f0 else 0

            embChannels = self.settings.modelSlots[self.currentSlot].embChannels
            audio_out = vc.pipeline(
                # self.hubert_model,
                self.embedder,
                self.onnx_session,
                sid,
                audio,
                f0_up_key,
                f0_method,
                self.index,
                self.feature,
                index_rate,
                if_f0,
                silence_front=self.settings.extraConvertSize
                / self.settings.modelSamplingRate,
                embChannels=embChannels,
            )
            result = audio_out * np.sqrt(vol)

        return result

    def _pyTorch_inference(self, data):
        if hasattr(self, "net_g") is False or self.net_g is None:
            print(
                "[Voice Changer] No pyTorch session.",
                hasattr(self, "net_g"),
                self.net_g,
            )
            raise NoModeLoadedException("pytorch")

        if self.settings.gpu < 0 or (self.gpu_num == 0 and self.mps_enabled is False):
            dev = torch.device("cpu")
        elif self.mps_enabled:
            dev = torch.device("mps")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        self.embedder = self.embedder.to(dev)
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
            f0_up_key = self.settings.tran
            f0_method = self.settings.f0Detector
            index_rate = self.settings.indexRatio
            if_f0 = 1 if self.settings.modelSlots[self.currentSlot].f0 else 0

            embChannels = self.settings.modelSlots[self.currentSlot].embChannels
            audio_out = vc.pipeline(
                # self.hubert_model,
                self.embedder,
                self.net_g,
                sid,
                audio,
                f0_up_key,
                f0_method,
                self.index,
                self.feature,
                index_rate,
                if_f0,
                silence_front=self.settings.extraConvertSize
                / self.settings.modelSamplingRate,
                embChannels=embChannels,
            )

            result = audio_out * np.sqrt(vol)

        return result

    def inference(self, data):
        if self.settings.modelSlotIndex < 0:
            print(
                "[Voice Changer] wait for loading model...",
                self.settings.modelSlotIndex,
                self.currentSlot,
            )
            raise NoModeLoadedException("model_common")

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

        print("---------- REMOVING ---------------")

        remove_path = os.path.join("RVC")
        sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("RVC" + os.path.sep) >= 0:
                    print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception:  # type:ignore
                # print(e)
                pass

    def export2onnx(self):
        if hasattr(self, "net_g") is False or self.net_g is None:
            print("[Voice Changer] export2onnx, No pyTorch session.")
            return {"status": "ng", "path": ""}

        pyTorchModelFile = self.settings.modelSlots[
            self.settings.modelSlotIndex
        ].pyTorchModelFile  # inference前にexportできるようにcurrentSlotではなくslot

        if pyTorchModelFile is None:
            print("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": ""}
        import voice_changer.RVC.export2onnx as onnxExporter

        output_file = os.path.splitext(os.path.basename(pyTorchModelFile))[0] + ".onnx"
        output_file_simple = (
            os.path.splitext(os.path.basename(pyTorchModelFile))[0] + "_simple.onnx"
        )
        output_path = os.path.join(TMP_DIR, output_file)
        output_path_simple = os.path.join(TMP_DIR, output_file_simple)
        print(
            "embChannels",
            self.settings.modelSlots[self.settings.modelSlotIndex].embChannels,
        )
        metadata = {
            "application": "VC_CLIENT",
            "version": "1",
            "modelType": self.settings.modelSlots[
                self.settings.modelSlotIndex
            ].modelType,
            "samplingRate": self.settings.modelSlots[
                self.settings.modelSlotIndex
            ].samplingRate,
            "f0": self.settings.modelSlots[self.settings.modelSlotIndex].f0,
            "embChannels": self.settings.modelSlots[
                self.settings.modelSlotIndex
            ].embChannels,
            "embedder": self.settings.modelSlots[self.settings.modelSlotIndex].embedder,
        }

        if torch.cuda.device_count() > 0:
            onnxExporter.export2onnx(
                pyTorchModelFile, output_path, output_path_simple, True, metadata
            )
        else:
            print(
                "[Voice Changer] Warning!!! onnx export with float32. maybe size is doubled."
            )
            onnxExporter.export2onnx(
                pyTorchModelFile, output_path, output_path_simple, False, metadata
            )

        return {
            "status": "ok",
            "path": f"/tmp/{output_file_simple}",
            "filename": output_file_simple,
        }

    def merge_models(self, request: str):
        print("[Voice Changer] MergeRequest:", request)
        req: MergeModelRequest = MergeModelRequest.from_json(request)
        merged = merge_model(req)
        targetSlot = 0
        if req.slot < 0:
            targetSlot = len(self.settings.modelSlots) - 1
        else:
            targetSlot = req.slot

        storeDir = os.path.join(UPLOAD_DIR, f"{targetSlot}")
        print("[Voice Changer] store merged model to:", storeDir)
        os.makedirs(storeDir, exist_ok=True)
        storeFile = os.path.join(storeDir, "merged.pth")
        torch.save(merged, storeFile)

        filePaths: FilePaths = FilePaths(
            pyTorchModelFilename=storeFile,
            configFilename=None,
            onnxModelFilename=None,
            featureFilename=None,
            indexFilename=None,
            clusterTorchModelFilename=None,
        )
        params = {"trans": req.defaultTrans}
        props: LoadModelParams = LoadModelParams(
            slot=targetSlot, isHalf=True, files=filePaths, params=json.dumps(params)
        )
        self.loadModel(props)
        self.prepareModel(targetSlot)
        self.settings.modelSlotIndex = targetSlot
        self.currentSlot = self.settings.modelSlotIndex
        # self.settings.tran = req.defaultTrans
