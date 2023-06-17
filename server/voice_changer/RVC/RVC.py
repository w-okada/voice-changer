import sys
import os
from dataclasses import asdict
from typing import cast
import numpy as np
import torch
import torchaudio
from data.ModelSlot import RVCModelSlot
from voice_changer.ModelSlotManager import ModelSlotManager


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

from voice_changer.RVC.modelMerger.MergeModel import merge_model
from voice_changer.RVC.modelMerger.MergeModelRequest import MergeModelRequest
from voice_changer.RVC.ModelSlotGenerator import (
    _setInfoByONNX,
    _setInfoByPytorch,
)
from voice_changer.RVC.RVCSettings import RVCSettings
from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from voice_changer.RVC.onnxExporter.export2onnx import export2onnx
from voice_changer.RVC.pitchExtractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.RVC.pipeline.PipelineGenerator import createPipeline
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.pipeline.Pipeline import Pipeline

from Exceptions import DeviceCannotSupportHalfPrecisionException, NoModeLoadedException
from const import (
    UPLOAD_DIR,
)
import shutil
import json


class RVC:
    initialLoad: bool = True
    settings: RVCSettings = RVCSettings()

    pipeline: Pipeline | None = None

    deviceManager = DeviceManager.get_instance()

    audio_buffer: AudioInOut | None = None
    prevVol: float = 0
    params: VoiceChangerParams
    currentSlot: int = 0
    needSwitch: bool = False

    def __init__(self, params: VoiceChangerParams):
        self.pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector)
        self.params = params
        EmbedderManager.initialize(params)
        print("[Voice Changer] RVC initialization: ", params)
        self.modelSlotManager = ModelSlotManager.get_instance(self.params.model_dir)

        # 起動時にスロットにモデルがある場合はロードしておく
        allSlots = self.modelSlotManager.getAllSlotInfo()
        availableIndex = -1
        for i, slot in enumerate(allSlots):
            if slot.modelFile is not None and slot.modelFile != "":
                availableIndex = i
                break
        if availableIndex >= 0:
            self.prepareModel(availableIndex)
            self.settings.modelSlotIndex = availableIndex
            self.switchModel(self.settings.modelSlotIndex)
            self.initialLoad = False

        self.prevVol = 0.0

    def moveToModelDir(self, file: str, dstDir: str):
        dst = os.path.join(dstDir, os.path.basename(file))
        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(file, dst)
        return dst

    def loadModel(self, props: LoadModelParams):
        target_slot_idx = props.slot
        params = props.params
        slotInfo: RVCModelSlot = RVCModelSlot()

        print("loadModel", params)
        slotInfo.modelFile = params["files"]["rvcModel"]
        slotInfo.indexFile = params["files"]["rvcIndex"] if "rvcIndex" in params["files"] else None

        slotInfo.defaultTune = params["defaultTune"]
        slotInfo.defaultIndexRatio = params["defaultIndexRatio"]
        slotInfo.defaultProtect = params["defaultProtect"]
        slotInfo.voiceChangerType = "RVC"
        slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")

        if slotInfo.isONNX:
            _setInfoByONNX(slotInfo)
        else:
            _setInfoByPytorch(slotInfo)

        # メタデータを見て、永続化モデルフォルダに移動させる
        # その際に、メタデータのファイル格納場所も書き換える
        slotDir = os.path.join(self.params.model_dir, str(target_slot_idx))
        os.makedirs(slotDir, exist_ok=True)
        slotInfo.modelFile = self.moveToModelDir(slotInfo.modelFile, slotDir)
        if slotInfo.indexFile is not None and len(slotInfo.indexFile) > 0:
            slotInfo.indexFile = self.moveToModelDir(slotInfo.indexFile, slotDir)
        if slotInfo.iconFile is not None and len(slotInfo.iconFile) > 0:
            slotInfo.iconFile = self.moveToModelDir(slotInfo.iconFile, slotDir)
        # json.dump(asdict(slotInfo), open(os.path.join(slotDir, "params.json"), "w"))
        self.modelSlotManager.save_model_slot(target_slot_idx, slotInfo)

        # 初回のみロード(起動時にスロットにモデルがあった場合はinitialLoadはFalseになっている)
        if self.initialLoad:
            self.prepareModel(target_slot_idx)
            self.settings.modelSlotIndex = target_slot_idx
            self.switchModel(self.settings.modelSlotIndex)
            self.initialLoad = False
        elif target_slot_idx == self.currentSlot:
            self.prepareModel(target_slot_idx)

        return self.get_info()

    def update_settings(self, key: str, val: int | float | str):
        if key in self.settings.intData:
            # 設定前処理
            val = cast(int, val)
            if key == "modelSlotIndex":
                if val < 0:
                    return True
                val = val % 1000  # Quick hack for same slot is selected
                allModelSlots = self.modelSlotManager.getAllSlotInfo()
                if allModelSlots[val].modelFile is None or allModelSlots[val].modelFile == "":
                    print("[Voice Changer] slot does not have model.")
                    return True
                self.prepareModel(val)

            # 設定
            setattr(self.settings, key, val)

            if key == "gpu":
                self.deviceManager.setForceTensor(False)
                self.prepareModel(self.settings.modelSlotIndex)

        elif key in self.settings.floatData:
            setattr(self.settings, key, float(val))
        elif key in self.settings.strData:
            setattr(self.settings, key, str(val))
            if key == "f0Detector" and self.pipeline is not None:
                pitchExtractor = PitchExtractorManager.getPitchExtractor(self.settings.f0Detector)
                self.pipeline.setPitchExtractor(pitchExtractor)
        else:
            return False
        return True

    def prepareModel(self, slot: int):
        if slot < 0:
            print("[Voice Changer] Prepare Model of slot skip:", slot)
            return self.get_info()
        allModelSlots = self.modelSlotManager.getAllSlotInfo()
        modelSlot = allModelSlots[slot]

        print("[Voice Changer] Prepare Model of slot:", slot)

        # pipelineの生成
        self.next_pipeline = createPipeline(modelSlot, self.settings.gpu, self.settings.f0Detector)

        # その他の設定
        self.next_trans = modelSlot.defaultTune
        self.next_index_ratio = modelSlot.defaultIndexRatio
        self.next_protect = modelSlot.defaultProtect
        self.next_samplingRate = modelSlot.samplingRate
        self.next_framework = "ONNX" if modelSlot.isONNX else "PyTorch"
        # self.needSwitch = True
        print("[Voice Changer] Prepare done.")
        self.switchModel(slot)
        return self.get_info()

    def switchModel(self, slot: int):
        print("[Voice Changer] Switching model..")
        self.pipeline = self.next_pipeline
        self.settings.tran = self.next_trans
        self.settings.indexRatio = self.next_index_ratio
        self.settings.protect = self.next_protect
        self.settings.modelSamplingRate = self.next_samplingRate
        self.settings.framework = self.next_framework

        # self.currentSlot = self.settings.modelSlotIndex # prepareModelから呼ばれるということはupdate_settingsの中で呼ばれるということなので、まだmodelSlotIndexは更新されていない
        self.currentSlot = slot
        print(
            "[Voice Changer] Switching model..done",
        )

    def get_info(self):
        data = asdict(self.settings)
        if self.pipeline is not None:
            pipelineInfo = self.pipeline.getPipelineInfo()
            data["pipelineInfo"] = pipelineInfo
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
        newData = newData.astype(np.float32) / 32768.0  # RVCのモデルのサンプリングレートで入ってきている。（extraDataLength, Crossfade等も同じSRで処理）(★１)

        if self.audio_buffer is not None:
            # 過去のデータに連結
            self.audio_buffer = np.concatenate([self.audio_buffer, newData], 0)
        else:
            self.audio_buffer = newData

        convertSize = inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize

        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (128 - (convertSize % 128))

        # バッファがたまっていない場合はzeroで補う
        if self.audio_buffer.shape[0] < convertSize:
            self.audio_buffer = np.concatenate([np.zeros([convertSize]), self.audio_buffer])

        convertOffset = -1 * convertSize
        self.audio_buffer = self.audio_buffer[convertOffset:]  # 変換対象の部分だけ抽出

        if self.pipeline is not None:
            device = self.pipeline.device
        else:
            device = torch.device("cpu")

        audio_buffer = torch.from_numpy(self.audio_buffer).to(device=device, dtype=torch.float32)

        # 出力部分だけ切り出して音量を確認。(TODO:段階的消音にする)
        cropOffset = -1 * (inputSize + crossfadeSize)
        cropEnd = -1 * (crossfadeSize)
        crop = audio_buffer[cropOffset:cropEnd]
        vol = torch.sqrt(torch.square(crop).mean()).detach().cpu().numpy()
        vol = max(vol, self.prevVol * 0.0)
        self.prevVol = vol

        return (audio_buffer, convertSize, vol)

    def inference(self, data):
        if self.settings.modelSlotIndex < 0:
            print(
                "[Voice Changer] wait for loading model...",
                self.settings.modelSlotIndex,
                self.currentSlot,
            )
            raise NoModeLoadedException("model_common")
        # if self.needSwitch:
        #     print(
        #         f"[Voice Changer] Switch model {self.currentSlot} -> {self.settings.modelSlotIndex}"
        #     )
        #     self.switchModel()
        #     self.needSwitch = False

        # half = self.deviceManager.halfPrecisionAvailable(self.settings.gpu)
        # half = self.pipeline.isHalf

        audio = data[0]
        convertSize = data[1]
        vol = data[2]

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        audio = torchaudio.functional.resample(audio, self.settings.modelSamplingRate, 16000, rolloff=0.99)
        repeat = 1 if self.settings.rvcQuality else 0
        sid = 0
        f0_up_key = self.settings.tran
        index_rate = self.settings.indexRatio
        protect = self.settings.protect

        # if_f0 = 1 if self.settings.modelSlots[self.currentSlot].f0 else 0
        # embOutputLayer = self.settings.modelSlots[self.currentSlot].embOutputLayer
        # useFinalProj = self.settings.modelSlots[self.currentSlot].useFinalProj

        if_f0 = 1 if self.modelSlotManager.get_slot_info(self.currentSlot).f0 else 0
        embOutputLayer = self.modelSlotManager.get_slot_info(self.currentSlot).embOutputLayer
        useFinalProj = self.modelSlotManager.get_slot_info(self.currentSlot).useFinalProj

        try:
            audio_out = self.pipeline.exec(
                sid,
                audio,
                f0_up_key,
                index_rate,
                if_f0,
                self.settings.extraConvertSize / self.settings.modelSamplingRate,  # extaraDataSizeの秒数。RVCのモデルのサンプリングレートで処理(★１)。
                embOutputLayer,
                useFinalProj,
                repeat,
                protect,
            )
            result = audio_out.detach().cpu().numpy() * np.sqrt(vol)

            return result
        except DeviceCannotSupportHalfPrecisionException as e:
            print("[Device Manager] Device cannot support half precision. Fallback to float....")
            self.deviceManager.setForceTensor(True)
            self.prepareModel(self.settings.modelSlotIndex)
            raise e

        return

    def __del__(self):
        del self.pipeline

        # print("---------- REMOVING ---------------")

        remove_path = os.path.join("RVC")
        sys.path = [x for x in sys.path if x.endswith(remove_path) is False]

        for key in list(sys.modules):
            val = sys.modules.get(key)
            try:
                file_path = val.__file__
                if file_path.find("RVC" + os.path.sep) >= 0:
                    # print("remove", key, file_path)
                    sys.modules.pop(key)
            except Exception:  # type:ignore
                # print(e)
                pass

    def export2onnx(self):
        allModelSlots = self.modelSlotManager.getAllSlotInfo()
        modelSlot = allModelSlots[self.settings.modelSlotIndex]

        if modelSlot.isONNX:
            print("[Voice Changer] export2onnx, No pyTorch filepath.")
            return {"status": "ng", "path": ""}

        output_file_simple = export2onnx(self.settings.gpu, modelSlot)
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
            # 最後尾のスロット番号を格納先とする。
            allModelSlots = self.modelSlotManager.getAllSlotInfo()
            targetSlot = len(allModelSlots) - 1
        else:
            targetSlot = req.slot

        # いったんは、アップロードフォルダに格納する。（歴史的経緯）
        # 後続のloadmodelを呼び出すことで永続化モデルフォルダに移動させられる。
        storeDir = os.path.join(UPLOAD_DIR, f"{targetSlot}")
        print("[Voice Changer] store merged model to:", storeDir)
        os.makedirs(storeDir, exist_ok=True)
        storeFile = os.path.join(storeDir, "merged.pth")
        torch.save(merged, storeFile)

        # loadmodelを呼び出して永続化モデルフォルダに移動させる。
        params = {
            "defaultTune": req.defaultTune,
            "defaultIndexRatio": req.defaultIndexRatio,
            "defaultProtect": req.defaultProtect,
            "sampleId": "",
            "files": {"rvcModel": storeFile},
        }
        props: LoadModelParams = LoadModelParams(slot=targetSlot, isHalf=True, params=params)
        self.loadModel(props)
        self.prepareModel(targetSlot)
        self.settings.modelSlotIndex = targetSlot
        self.currentSlot = self.settings.modelSlotIndex

    def update_model_default(self):
        # {"slot":9,"key":"name","val":"dogsdododg"}
        self.modelSlotManager.update_model_info(
            json.dumps(
                {
                    "slot": self.currentSlot,
                    "key": "defaultTune",
                    "val": self.settings.tran,
                }
            )
        )
        self.modelSlotManager.update_model_info(
            json.dumps(
                {
                    "slot": self.currentSlot,
                    "key": "defaultIndexRatio",
                    "val": self.settings.indexRatio,
                }
            )
        )
        self.modelSlotManager.update_model_info(
            json.dumps(
                {
                    "slot": self.currentSlot,
                    "key": "defaultProtect",
                    "val": self.settings.protect,
                }
            )
        )

    def update_model_info(self, newData: str):
        self.modelSlotManager.update_model_info(newData)

    def upload_model_assets(self, params: str):
        self.modelSlotManager.store_model_assets(params)
