import sys
import os

from voice_changer.RVC.pitchExtractor.PitchExtractor import PitchExtractor
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager

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
import json
import resampy
from voice_changer.RVC.MergeModel import merge_model
from voice_changer.RVC.MergeModelRequest import MergeModelRequest
from voice_changer.RVC.ModelSlotGenerator import generateModelSlot
from Exceptions import NoModeLoadedException
from voice_changer.RVC.RVCSettings import RVCSettings
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from voice_changer.RVC.inferencer.InferencerManager import InferencerManager
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

from const import TMP_DIR, UPLOAD_DIR


from voice_changer.RVC.custom_vc_infer_pipeline import VC

providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


class RVC:
    audio_buffer: AudioInOut | None = None
    embedder: Embedder | None = None
    inferencer: Inferencer | None = None
    pitchExtractor: PitchExtractor | None = None

    def __init__(self, params: VoiceChangerParams):
        self.initialLoad = True
        self.settings = RVCSettings()
        self.pitchExtractor = PitchExtractorManager.getPitchExtractor(
            self.settings.f0Detector
        )

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
        loadModelはスロットへのエントリ(推論向けにはロードしない)。
        例外的に、まだ一つも推論向けにロードされていない場合と稼働中スロットの場合は、ロードする。
        """
        self.is_half = props.isHalf
        target_slot_idx = props.slot
        params_str = props.params
        params = json.loads(params_str)

        modelSlot = generateModelSlot(props.files, params)
        self.settings.modelSlots[target_slot_idx] = modelSlot
        print(
            f"[Voice Changer] RVC new model is uploaded,{target_slot_idx}",
            asdict(modelSlot),
        )

        # 初回のみロード
        if self.initialLoad or target_slot_idx == self.currentSlot:
            self.prepareModel(target_slot_idx)
            self.settings.modelSlotIndex = target_slot_idx
            # self.currentSlot = self.settings.modelSlotIndex
            self.switchModel()
            self.initialLoad = False

        return self.get_info()

    def _getDevice(self):
        if self.settings.gpu < 0 or (self.gpu_num == 0 and self.mps_enabled is False):
            dev = torch.device("cpu")
        elif self.mps_enabled:
            dev = torch.device("mps")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)
        return dev

    def prepareModel(self, slot: int):
        if slot < 0:
            return self.get_info()
        print("[Voice Changer] Prepare Model of slot:", slot)
        modelSlot = self.settings.modelSlots[slot]
        filename = (
            modelSlot.onnxModelFile if modelSlot.isONNX else modelSlot.pyTorchModelFile
        )
        dev = self._getDevice()

        # Inferencerのロード
        inferencer = InferencerManager.getInferencer(
            modelSlot.modelType,
            filename,
            self.settings.isHalf,
            dev,
        )
        self.next_inferencer = inferencer

        # Indexのロード
        print("[Voice Changer] Loading index...")
        if modelSlot.featureFile is not None and modelSlot.indexFile is not None:
            if (
                os.path.exists(modelSlot.featureFile) is True
                and os.path.exists(modelSlot.indexFile) is True
            ):
                try:
                    self.next_index = faiss.read_index(modelSlot.indexFile)
                    self.next_feature = np.load(modelSlot.featureFile)
                except:
                    print("[Voice Changer] load index failed. Use no index.")
                    traceback.print_exc()
                    self.next_index = self.next_feature = None
            else:
                print("[Voice Changer] Index file is not found. Use no index.")
                self.next_index = self.next_feature = None
        else:
            self.next_index = self.next_feature = None

        self.next_trans = modelSlot.defaultTrans
        self.next_samplingRate = modelSlot.samplingRate
        self.next_embedder = modelSlot.embedder
        self.next_framework = "ONNX" if modelSlot.isONNX else "PyTorch"
        print("[Voice Changer] Prepare done.")
        return self.get_info()

    def switchModel(self):
        print("[Voice Changer] Switching model..")
        if self.settings.gpu < 0 or (self.gpu_num == 0 and self.mps_enabled is False):
            dev = torch.device("cpu")
        elif self.mps_enabled:
            dev = torch.device("mps")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        # embedderはモデルによらず再利用できる可能性が高いので、Switchのタイミングでこちらで取得
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

        self.inferencer = self.next_inferencer
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
        # if key == "onnxExecutionProvider" and self.onnx_session is not None:
        #     if val == "CUDAExecutionProvider":
        #         if self.settings.gpu < 0 or self.settings.gpu >= self.gpu_num:
        #             self.settings.gpu = 0
        #         provider_options = [{"device_id": self.settings.gpu}]
        #         self.onnx_session.set_providers(
        #             providers=[val], provider_options=provider_options
        #         )
        #         if hasattr(self, "hubert_onnx"):
        #             self.hubert_onnx.set_providers(
        #                 providers=[val], provider_options=provider_options
        #             )
        #     else:
        #         self.onnx_session.set_providers(providers=[val])
        #         if hasattr(self, "hubert_onnx"):
        #             self.hubert_onnx.set_providers(providers=[val])
        # elif key == "onnxExecutionProvider" and self.onnx_session is None:
        #     print("Onnx is not enabled. Please load model.")
        #     return False
        if key in self.settings.intData:
            val = cast(int, val)
            # if (
            #     key == "gpu"
            #     and val >= 0
            #     and val < self.gpu_num
            #     and self.onnx_session is not None
            # ):
            #     providers = self.onnx_session.get_providers()
            #     print("Providers:", providers)
            #     if "CUDAExecutionProvider" in providers:
            #         provider_options = [{"device_id": self.settings.gpu}]
            #         self.onnx_session.set_providers(
            #             providers=["CUDAExecutionProvider"],
            #             provider_options=provider_options,
            #         )
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

        # data["onnxExecutionProviders"] = (
        #     self.onnx_session.get_providers() if self.onnx_session is not None else []
        # )
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
            vc = VC(
                self.settings.modelSamplingRate,
                torch.device("cuda:0"),
                self.is_half,
                repeat,
            )
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
                self.pitchExtractor,
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
        # if hasattr(self, "net_g") is False or self.net_g is None:
        #     print(
        #         "[Voice Changer] No pyTorch session.",
        #         hasattr(self, "net_g"),
        #         self.net_g,
        #     )
        #     raise NoModeLoadedException("pytorch")

        if self.settings.gpu < 0 or (self.gpu_num == 0 and self.mps_enabled is False):
            dev = torch.device("cpu")
        elif self.mps_enabled:
            dev = torch.device("mps")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        self.embedder = self.embedder.to(dev)
        self.inferencer = self.inferencer.to(dev)

        # self.embedder.printDevice()
        # self.inferencer.printDevice()

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
                self.embedder,
                self.inferencer,
                self.pitchExtractor,
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
