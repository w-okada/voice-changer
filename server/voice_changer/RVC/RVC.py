import sys
import os
from Exceptions import NoModeLoadedException
from voice_changer.RVC.ModelSlot import ModelSlot
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager

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

from const import UPLOAD_DIR, EnumEmbedderTypes


from voice_changer.RVC.custom_vc_infer_pipeline import VC

providers = [
    "OpenVINOExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


class RVC:
    initialLoad: bool = True
    settings: RVCSettings = RVCSettings()

    embedder: Embedder | None = None
    inferencer: Inferencer | None = None

    pitchExtractor: PitchExtractor | None = None
    deviceManager = DeviceManager.get_instance()

    audio_buffer: AudioInOut | None = None
    prevVol: float = 0
    params: VoiceChangerParams
    currentSlot: int = -1
    needSwitch: bool = False

    def __init__(self, params: VoiceChangerParams):
        self.pitchExtractor = PitchExtractorManager.getPitchExtractor(
            self.settings.f0Detector
        )
        self.params = params
        print("RVC initialization: ", params)

    def loadModel(self, props: LoadModelParams):
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
        if self.initialLoad:
            self.prepareModel(target_slot_idx)
            self.settings.modelSlotIndex = target_slot_idx
            self.switchModel()
            self.initialLoad = False
        elif target_slot_idx == self.currentSlot:
            self.prepareModel(target_slot_idx)
            self.needSwitch = True

        return self.get_info()

    def createPipeline(self, modelSlot: ModelSlot):
        dev = self.deviceManager.getDevice(self.settings.gpu)
        half = self.deviceManager.halfPrecisionAvailable(self.settings.gpu)
        # ファイル名特定(Inferencer)
        inferencerFilename = (
            modelSlot.onnxModelFile if modelSlot.isONNX else modelSlot.pyTorchModelFile
        )
        # ファイル名特定(embedder)
        if modelSlot.embedder == EnumEmbedderTypes.hubert:
            emmbedderFilename = self.params.hubert_base
        elif modelSlot.embedder == EnumEmbedderTypes.contentvec:
            emmbedderFilename = self.params.content_vec_500
        elif modelSlot.embedder == EnumEmbedderTypes.hubert_jp:
            emmbedderFilename = self.params.hubert_base_jp
        else:
            raise RuntimeError(
                "[Voice Changer] Exception loading embedder failed. unknwon type:",
                modelSlot.embedder,
            )

        # Inferencer 生成
        try:
            inferencer = InferencerManager.getInferencer(
                modelSlot.modelType,
                inferencerFilename,
                half,
                dev,
            )
        except Exception as e:
            print("[Voice Changer] exception! loading inferencer", e)
            traceback.print_exc()

        # Embedder 生成
        try:
            embedder = EmbedderManager.getEmbedder(
                modelSlot.embedder,
                emmbedderFilename,
                half,
                dev,
            )
        except Exception as e:
            print("[Voice Changer]  exception! loading embedder", e)
            traceback.print_exc()

        return inferencer, embedder

    def loadIndex(self, modelSlot: ModelSlot):
        # Indexのロード
        print("[Voice Changer] Loading index...")
        # ファイル指定がない場合はNone
        if modelSlot.featureFile is None or modelSlot.indexFile is None:
            return None, None

        # ファイル指定があってもファイルがない場合はNone
        if (
            os.path.exists(modelSlot.featureFile) is not True
            or os.path.exists(modelSlot.indexFile) is not True
        ):
            return None, None

        try:
            index = faiss.read_index(modelSlot.indexFile)
            feature = np.load(modelSlot.featureFile)
        except:
            print("[Voice Changer] load index failed. Use no index.")
            traceback.print_exc()
            return None, None

        return index, feature

    def prepareModel(self, slot: int):
        if slot < 0:
            return self.get_info()
        modelSlot = self.settings.modelSlots[slot]
        inferencerFilename = (
            modelSlot.onnxModelFile if modelSlot.isONNX else modelSlot.pyTorchModelFile
        )
        if inferencerFilename == "":
            return self.get_info()

        print("[Voice Changer] Prepare Model of slot:", slot)

        # Inferencer, embedderのロード
        inferencer, embedder = self.createPipeline(modelSlot)

        self.next_inferencer = inferencer
        self.next_embedder = embedder

        # Indexのロード
        index, feature = self.loadIndex(modelSlot)
        self.next_index = index
        self.next_feature = feature

        # その他の設定
        self.next_trans = modelSlot.defaultTrans
        self.next_samplingRate = modelSlot.samplingRate
        self.next_framework = "ONNX" if modelSlot.isONNX else "PyTorch"
        self.needSwitch = True
        print("[Voice Changer] Prepare done.")
        return self.get_info()

    def switchModel(self):
        print("[Voice Changer] Switching model..")
        self.embedder = self.next_embedder
        self.inferencer = self.next_inferencer
        self.feature = self.next_feature
        self.index = self.next_index
        self.settings.tran = self.next_trans
        self.settings.modelSamplingRate = self.next_samplingRate
        self.settings.framework = self.next_framework

        print(
            "[Voice Changer] Switching model..done",
        )

    def update_settings(self, key: str, val: int | float | str):
        if key in self.settings.intData:
            # 設定前処理
            val = cast(int, val)
            if key == "modelSlotIndex":
                if val < 0:
                    return True
                val = val % 1000  # Quick hack for same slot is selected
                self.prepareModel(val)
                self.needSwitch = True

            # 設定
            setattr(self.settings, key, val)

            if key == "gpu" and self.embedder is not None:
                dev = self.deviceManager.getDevice(val)
                half = self.deviceManager.halfPrecisionAvailable(val)

                # half-precisionの使用可否が変わるときは作り直し
                if (
                    self.inferencer is not None
                    and self.inferencer.isHalf == half
                    and self.embedder.isHalf == half
                ):
                    print(
                        "USE EXSISTING PIPELINE",
                        half,
                    )
                    self.embedder.setDevice(dev)
                    self.inferencer.setDevice(dev)
                else:
                    print("CHAGE TO NEW PIPELINE", half)
                    self.prepareModel(self.settings.modelSlotIndex)
        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
        else:
            return False
        return True

    def get_info(self):
        data = asdict(self.settings)
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

    def inference(self, data):
        if self.settings.modelSlotIndex < 0:
            print(
                "[Voice Changer] wait for loading model...",
                self.settings.modelSlotIndex,
                self.currentSlot,
            )
            raise NoModeLoadedException("model_common")
        if self.needSwitch:
            print(
                f"[Voice Changer] Switch model {self.currentSlot} -> {self.settings.modelSlotIndex}"
            )
            self.currentSlot = self.settings.modelSlotIndex
            self.switchModel()
            self.needSwitch = False

        dev = self.deviceManager.getDevice(self.settings.gpu)
        half = self.deviceManager.halfPrecisionAvailable(self.settings.gpu)

        # self.embedder = self.embedder.setDevice(dev)
        # self.inferencer = self.inferencer.setDevice(dev)

        audio = data[0]
        convertSize = data[1]
        vol = data[2]

        audio = resampy.resample(audio, self.settings.modelSamplingRate, 16000)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        repeat = 3 if half else 1
        repeat *= self.settings.rvcQuality  # 0 or 3
        vc = VC(self.settings.modelSamplingRate, dev, half, repeat)
        sid = 0
        f0_up_key = self.settings.tran
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
            self.index,
            self.feature,
            index_rate,
            if_f0,
            silence_front=self.settings.extraConvertSize
            / self.settings.modelSamplingRate,
            embChannels=embChannels,
        )

        result = audio_out * np.sqrt(vol)

        # if self.settings.framework == "ONNX":
        #     audio = self._onnx_inference(data)
        # else:
        #     audio = self._pyTorch_inference(data)

        return result

    def __del__(self):
        del self.inferencer
        del self.embedder

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
        modelSlot = self.settings.modelSlots[self.settings.modelSlotIndex]
        pyTorchModelFile = modelSlot.pyTorchModelFile

        # PyTorchのファイルが存在しない場合はエラーを返す
        if pyTorchModelFile is None or pyTorchModelFile == "":
            print("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": ""}

        import voice_changer.RVC.export2onnx as onnxExporter

        output_file_simple = onnxExporter.export2onnx(self.settings.gpu, modelSlot)
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
