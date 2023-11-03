import json
import os
import sys
import shutil
import threading
import numpy as np
from downloader.SampleDownloader import downloadSample, getSampleInfos
from mods.log_control import VoiceChangaerLogger
from voice_changer.Local.ServerDevice import ServerDevice, ServerDeviceCallbacks
from voice_changer.ModelSlotManager import ModelSlotManager
from voice_changer.RVC.RVCModelMerger import RVCModelMerger
from voice_changer.VoiceChanger import VoiceChanger
from const import STORED_SETTING_FILE, UPLOAD_DIR, StaticSlot
from voice_changer.VoiceChangerV2 import VoiceChangerV2
from voice_changer.utils.LoadModelParams import LoadModelParamFile, LoadModelParams
from voice_changer.utils.ModelMerger import MergeElement, ModelMergerRequest
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from dataclasses import dataclass, asdict, field
import torch

# import threading
from typing import Callable
from typing import Any
import re

logger = VoiceChangaerLogger.get_instance().getLogger()


@dataclass()
class GPUInfo:
    id: int
    name: str
    memory: int


@dataclass()
class VoiceChangerManagerSettings:
    modelSlotIndex: int | StaticSlot = -1
    passThrough: bool = False  # 0: off, 1: on
    # ↓mutableな物だけ列挙
    boolData: list[str] = field(default_factory=lambda: ["passThrough"])
    intData: list[str] = field(
        default_factory=lambda: [
            "modelSlotIndex",
        ]
    )


class VoiceChangerManager(ServerDeviceCallbacks):
    _instance = None

    ############################
    # ServerDeviceCallbacks
    ############################
    def on_request(self, unpackedData: AudioInOut):
        return self.changeVoice(unpackedData)

    def emitTo(self, performance: list[float]):
        self.emitToFunc(performance)

    def get_processing_sampling_rate(self):
        return self.voiceChanger.get_processing_sampling_rate()

    def setInputSamplingRate(self, sr: int):
        self.voiceChanger.setInputSampleRate(sr)

    def setOutputSamplingRate(self, sr: int):
        self.voiceChanger.setOutputSampleRate(sr)

    ############################
    # VoiceChangerManager
    ############################
    def __init__(self, params: VoiceChangerParams):
        logger.info("[Voice Changer] VoiceChangerManager initializing...")
        self.params = params
        self.voiceChanger: VoiceChanger = None
        self.settings: VoiceChangerManagerSettings = VoiceChangerManagerSettings()

        self.modelSlotManager = ModelSlotManager.get_instance(self.params.model_dir)
        # スタティックな情報を収集
        self.gpus: list[GPUInfo] = self._get_gpuInfos()

        self.serverDevice = ServerDevice(self)

        thread = threading.Thread(target=self.serverDevice.start, args=())
        thread.start()

        # 設定保存用情報
        self.stored_setting: dict[str, str | int | float] = {}
        if os.path.exists(STORED_SETTING_FILE):
            self.stored_setting = json.load(open(STORED_SETTING_FILE, "r", encoding="utf-8"))
        if "modelSlotIndex" in self.stored_setting:
            self.update_settings("modelSlotIndex", self.stored_setting["modelSlotIndex"])
        if "gpu" not in self.stored_setting:
            self.update_settings("gpu", 0)
        # for key, val in self.stored_setting.items():
        #     self.update_settings(key, val)
        logger.info("[Voice Changer] VoiceChangerManager initializing... done.")

    def store_setting(self, key: str, val: str | int | float):
        saveItemForServerDevice = ["enableServerAudio", "serverAudioSampleRate", "serverInputDeviceId", "serverOutputDeviceId", "serverMonitorDeviceId", "serverReadChunkSize", "serverInputAudioGain", "serverOutputAudioGain"]
        saveItemForVoiceChanger = ["crossFadeOffsetRate", "crossFadeEndRate", "crossFadeOverlapSize"]
        saveItemForVoiceChangerManager = ["modelSlotIndex"]
        saveItemForRVC = ["extraConvertSize", "gpu", "silentThreshold"]
        saveItemForAllVoiceChanger = ["f0Detector"]  # 設定されたf0DetectorがVCに存在しない値の場合はデフォルトに落ちるように実装すること

        saveItem = []
        saveItem.extend(saveItemForServerDevice)
        saveItem.extend(saveItemForVoiceChanger)
        saveItem.extend(saveItemForVoiceChangerManager)
        saveItem.extend(saveItemForRVC)
        saveItem.extend(saveItemForAllVoiceChanger)
        if key in saveItem:
            self.stored_setting[key] = val
            json.dump(self.stored_setting, open(STORED_SETTING_FILE, "w"))

    def _get_gpuInfos(self):
        devCount = torch.cuda.device_count()
        gpus = []
        for id in range(devCount):
            name = torch.cuda.get_device_name(id)
            memory = torch.cuda.get_device_properties(id).total_memory
            gpu = {"id": id, "name": name, "memory": memory}
            gpus.append(gpu)
        return gpus

    @classmethod
    def get_instance(cls, params: VoiceChangerParams):
        if cls._instance is None:
            cls._instance = cls(params)
        return cls._instance

    def loadModel(self, params: LoadModelParams):
        if params.isSampleMode:
            # サンプルダウンロード
            logger.info(f"[Voice Changer] sample download...., {params}")
            downloadSample(self.params.sample_mode, params.sampleId, self.params.model_dir, params.slot, params.params)
            self.modelSlotManager.getAllSlotInfo(reload=True)
            info = {"status": "OK"}
            return info
        else:
            # アップローダ
            # ファイルをslotにコピー
            slotDir = os.path.join(
                self.params.model_dir,
                str(params.slot),
            )
            if os.path.isdir(slotDir):
                shutil.rmtree(slotDir)

            for file in params.files:
                logger.info(f"FILE: {file}")
                srcPath = os.path.join(UPLOAD_DIR, file.dir, file.name)
                dstDir = os.path.join(
                    self.params.model_dir,
                    str(params.slot),
                    file.dir,
                )
                dstPath = os.path.join(dstDir, file.name)
                os.makedirs(dstDir, exist_ok=True)
                logger.info(f"move to {srcPath} -> {dstPath}")
                shutil.move(srcPath, dstPath)
                file.name = os.path.basename(dstPath)

            # メタデータ作成(各VCで定義)
            if params.voiceChangerType == "RVC":
                from voice_changer.RVC.RVCModelSlotGenerator import RVCModelSlotGenerator  # 起動時にインポートするとパラメータが取れない。

                slotInfo = RVCModelSlotGenerator.loadModel(params)
                self.modelSlotManager.save_model_slot(params.slot, slotInfo)
            elif params.voiceChangerType == "MMVCv13":
                from voice_changer.MMVCv13.MMVCv13ModelSlotGenerator import MMVCv13ModelSlotGenerator

                slotInfo = MMVCv13ModelSlotGenerator.loadModel(params)
                self.modelSlotManager.save_model_slot(params.slot, slotInfo)
            elif params.voiceChangerType == "MMVCv15":
                from voice_changer.MMVCv15.MMVCv15ModelSlotGenerator import MMVCv15ModelSlotGenerator

                slotInfo = MMVCv15ModelSlotGenerator.loadModel(params)
                self.modelSlotManager.save_model_slot(params.slot, slotInfo)
            elif params.voiceChangerType == "so-vits-svc-40":
                from voice_changer.SoVitsSvc40.SoVitsSvc40ModelSlotGenerator import SoVitsSvc40ModelSlotGenerator

                slotInfo = SoVitsSvc40ModelSlotGenerator.loadModel(params)
                self.modelSlotManager.save_model_slot(params.slot, slotInfo)
            elif params.voiceChangerType == "DDSP-SVC":
                from voice_changer.DDSP_SVC.DDSP_SVCModelSlotGenerator import DDSP_SVCModelSlotGenerator

                slotInfo = DDSP_SVCModelSlotGenerator.loadModel(params)
                self.modelSlotManager.save_model_slot(params.slot, slotInfo)
            elif params.voiceChangerType == "Diffusion-SVC":
                from voice_changer.DiffusionSVC.DiffusionSVCModelSlotGenerator import DiffusionSVCModelSlotGenerator

                slotInfo = DiffusionSVCModelSlotGenerator.loadModel(params)
                self.modelSlotManager.save_model_slot(params.slot, slotInfo)
            elif params.voiceChangerType == "Beatrice":
                from voice_changer.Beatrice.BeatriceModelSlotGenerator import BeatriceModelSlotGenerator

                slotInfo = BeatriceModelSlotGenerator.loadModel(params)
                self.modelSlotManager.save_model_slot(params.slot, slotInfo)
            logger.info(f"params, {params}")

    def get_info(self):
        data = asdict(self.settings)
        data["gpus"] = self.gpus
        data["modelSlots"] = self.modelSlotManager.getAllSlotInfo(reload=True)
        data["sampleModels"] = getSampleInfos(self.params.sample_mode)
        data["python"] = sys.version
        data["voiceChangerParams"] = self.params

        data["status"] = "OK"

        info = self.serverDevice.get_info()
        data.update(info)

        if self.voiceChanger is not None:
            info = self.voiceChanger.get_info()
            data.update(info)

        return data

    def get_performance(self):
        if hasattr(self, "voiceChanger"):
            info = self.voiceChanger.get_performance()
            return info
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

    def generateVoiceChanger(self, val: int | StaticSlot):
        slotInfo = self.modelSlotManager.get_slot_info(val)
        if slotInfo is None:
            logger.info(f"[Voice Changer] model slot is not found {val}")
            return
        elif slotInfo.voiceChangerType == "RVC":
            logger.info("................RVC")
            # from voice_changer.RVC.RVC import RVC

            # self.voiceChangerModel = RVC(self.params, slotInfo)
            # self.voiceChanger = VoiceChanger(self.params)
            # self.voiceChanger.setModel(self.voiceChangerModel)

            from voice_changer.RVC.RVCr2 import RVCr2

            self.voiceChangerModel = RVCr2(self.params, slotInfo)
            self.voiceChanger = VoiceChangerV2(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)

        elif slotInfo.voiceChangerType == "MMVCv13":
            logger.info("................MMVCv13")
            from voice_changer.MMVCv13.MMVCv13 import MMVCv13

            self.voiceChangerModel = MMVCv13(slotInfo)
            self.voiceChanger = VoiceChanger(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)
        elif slotInfo.voiceChangerType == "MMVCv15":
            logger.info("................MMVCv15")
            from voice_changer.MMVCv15.MMVCv15 import MMVCv15

            self.voiceChangerModel = MMVCv15(slotInfo)
            self.voiceChanger = VoiceChanger(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)
        elif slotInfo.voiceChangerType == "so-vits-svc-40":
            logger.info("................so-vits-svc-40")
            from voice_changer.SoVitsSvc40.SoVitsSvc40 import SoVitsSvc40

            self.voiceChangerModel = SoVitsSvc40(self.params, slotInfo)
            self.voiceChanger = VoiceChanger(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)
        elif slotInfo.voiceChangerType == "DDSP-SVC":
            logger.info("................DDSP-SVC")
            from voice_changer.DDSP_SVC.DDSP_SVC import DDSP_SVC

            self.voiceChangerModel = DDSP_SVC(self.params, slotInfo)
            self.voiceChanger = VoiceChanger(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)
        elif slotInfo.voiceChangerType == "Diffusion-SVC":
            logger.info("................Diffusion-SVC")
            from voice_changer.DiffusionSVC.DiffusionSVC import DiffusionSVC

            self.voiceChangerModel = DiffusionSVC(self.params, slotInfo)
            self.voiceChanger = VoiceChangerV2(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)
        elif slotInfo.voiceChangerType == "Beatrice":
            logger.info("................Beatrice")
            from voice_changer.Beatrice.Beatrice import Beatrice

            if val == "Beatrice-JVS":
                self.voiceChangerModel = Beatrice(self.params, slotInfo, static=True)
            else:
                self.voiceChangerModel = Beatrice(self.params, slotInfo)
            self.voiceChanger = VoiceChangerV2(self.params)
            self.voiceChanger.setModel(self.voiceChangerModel)
        else:
            logger.info(f"[Voice Changer] unknown voice changer model: {slotInfo.voiceChangerType}")
            if hasattr(self, "voiceChangerModel"):
                del self.voiceChangerModel
            return

    def update_settings(self, key: str, val: str | int | float | bool):
        self.store_setting(key, val)

        if key in self.settings.boolData:
            if val == "true":
                newVal = True
            elif val == "false":
                newVal = False
            setattr(self.settings, key, newVal)
        elif key in self.settings.intData:
            if key == "modelSlotIndex":
                try:
                    newVal = int(val)
                    newVal = newVal % 1000
                except:
                    newVal = re.sub("^\d+", "", val)  # 先頭の数字を取り除く。
                logger.info(f"[Voice Changer] model slot is changed {self.settings.modelSlotIndex} -> {newVal}")
                self.generateVoiceChanger(newVal)
                # キャッシュ設定の反映
                for k, v in self.stored_setting.items():
                    if k != "modelSlotIndex":
                        self.update_settings(k, v)
            else:
                newVal = int(val)

            setattr(self.settings, key, newVal)

        self.serverDevice.update_settings(key, val)
        if self.voiceChanger is not None:
            self.voiceChanger.update_settings(key, val)

        return self.get_info()

    def changeVoice(self, receivedData: AudioInOut):
        if self.settings.passThrough is True:  # パススルー
            return receivedData, []

        if hasattr(self, "voiceChanger") is True:
            return self.voiceChanger.on_request(receivedData)
        else:
            logger.info("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16), []

    def export2onnx(self):
        return self.voiceChanger.export2onnx()

    def merge_models(self, request: str):
        # self.voiceChanger.merge_models(request)
        req = json.loads(request)
        req = ModelMergerRequest(**req)
        req.files = [MergeElement(**f) for f in req.files]
        slot = len(self.modelSlotManager.getAllSlotInfo()) - 1
        if req.voiceChangerType == "RVC":
            merged = RVCModelMerger.merge_models(self.params, req, slot)
            loadParam = LoadModelParams(voiceChangerType="RVC", slot=slot, isSampleMode=False, sampleId="", files=[LoadModelParamFile(name=os.path.basename(merged), kind="rvcModel", dir="")], params={})
            self.loadModel(loadParam)
        return self.get_info()

    def setEmitTo(self, emitTo: Callable[[Any], None]):
        self.emitToFunc = emitTo

    def update_model_default(self):
        # self.voiceChanger.update_model_default()
        current_settings = self.voiceChangerModel.get_model_current()
        for current_setting in current_settings:
            current_setting["slot"] = self.settings.modelSlotIndex
            self.modelSlotManager.update_model_info(json.dumps(current_setting))
        return self.get_info()

    def update_model_info(self, newData: str):
        # self.voiceChanger.update_model_info(newData)
        self.modelSlotManager.update_model_info(newData)
        return self.get_info()

    def upload_model_assets(self, params: str):
        # self.voiceChanger.upload_model_assets(params)
        self.modelSlotManager.store_model_assets(params)
        return self.get_info()
