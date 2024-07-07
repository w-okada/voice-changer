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
from const import STORED_SETTING_FILE, UPLOAD_DIR
from voice_changer.VoiceChangerSettings import VoiceChangerSettings
from voice_changer.VoiceChangerV2 import VoiceChangerV2
from voice_changer.utils.LoadModelParams import LoadModelParamFile, LoadModelParams
from voice_changer.utils.ModelMerger import MergeElement, ModelMergerRequest
from voice_changer.utils.VoiceChangerModel import AudioInOut
from settings import ServerSettings
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from Exceptions import (
    NoModeLoadedException,
    PipelineNotInitializedException,
    VoiceChangerIsNotSelectedException,
)

# import threading
from typing import Callable, Any

logger = VoiceChangaerLogger.get_instance().getLogger()


class VoiceChangerManager(ServerDeviceCallbacks):
    _instance = None

    ############################
    # ServerDeviceCallbacks
    ############################
    def on_request(self, unpackedData: AudioInOut):
        return self.changeVoice(unpackedData)

    def emitTo(self, performance: list[float]):
        self.emitToFunc(performance)

    ############################
    # VoiceChangerManager
    ############################
    def __init__(self, params: ServerSettings):
        logger.info("[Voice Changer] VoiceChangerManager initializing...")
        self.params = params
        self.voiceChanger: VoiceChangerV2 = None
        self.voiceChangerModel = None

        self.modelSlotManager = ModelSlotManager.get_instance(self.params.model_dir)
        # スタティックな情報を収集

        self.settings = VoiceChangerSettings()
        try:
            with open(STORED_SETTING_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            self.settings.set_properties(settings)
        except:
            pass

        self.device_manager = DeviceManager.get_instance()
        self.devices = self.device_manager.list_devices()
        self.device_manager.initialize(self.settings.gpu, self.settings.forceFp32)

        self.serverDevice = ServerDevice(self, self.settings)

        thread = threading.Thread(target=self.serverDevice.start, args=())
        thread.start()

        logger.info("[Voice Changer] VoiceChangerManager initializing... done.")

        # Initialize the voice changer
        self.initialize(self.settings.modelSlotIndex)

    def store_setting(self):
        with open(STORED_SETTING_FILE, "w") as f:
            json.dump(self.settings.to_dict_stateless(), f)

    @classmethod
    def get_instance(cls, params: ServerSettings):
        if cls._instance is None:
            cls._instance = cls(params)
        return cls._instance

    async def load_model(self, params: LoadModelParams):
        if params.isSampleMode:
            # サンプルダウンロード
            logger.info(f"[Voice Changer] sample download...., {params}")
            await downloadSample(self.params.sample_mode, params.sampleId, self.params.model_dir, params.slot, params.params)
            self.modelSlotManager.getAllSlotInfo(reload=True)
            info = {"status": "OK"}
            return info

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

            slotInfo = RVCModelSlotGenerator.load_model(params)
            self.modelSlotManager.save_model_slot(params.slot, slotInfo)

        logger.info(f"params, {params}")

    def get_info(self):
        data = self.settings.to_dict()
        data["gpus"] = self.devices
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

    def initialize(self, val: int):
        slotInfo = self.modelSlotManager.get_slot_info(val)
        if slotInfo is None:
            logger.info(f"[Voice Changer] model slot is not found {val}")
            return

        if self.voiceChangerModel is not None and slotInfo.voiceChangerType == self.voiceChangerModel.voiceChangerType:
            self.voiceChangerModel.set_slot_info(slotInfo)
            self.voiceChanger.set_model(self.voiceChangerModel)
            self.voiceChangerModel.initialize()
            return

        if slotInfo.voiceChangerType == "RVC":
            logger.info("................RVC")
            from voice_changer.RVC.RVCr2 import RVCr2

            self.voiceChangerModel = RVCr2(self.params, slotInfo, self.settings)
            self.voiceChanger = VoiceChangerV2(self.params, self.settings)
            self.voiceChanger.set_model(self.voiceChangerModel)
        else:
            logger.info(f"[Voice Changer] unknown voice changer model: {slotInfo.voiceChangerType}")

    def update_settings(self, key: str, val: Any):
        print("[Voice Changer] update configuration:", key, val)
        error, old_value = self.settings.set_property(key, val)
        if error:
            return self.get_info()
        # TODO: This is required to get type-casted setting. But maybe this should be done prior to setting.
        val = self.settings.get_property(key)
        if old_value == val:
            return self.get_info()
        # TODO: Storing settings on each change is suboptimal. Maybe timed autosave?
        self.store_setting()

        if key == "modelSlotIndex":
            logger.info(f"[Voice Changer] Model slot is changed {old_value} -> {val}")
            self.initialize(val)
        elif key == 'gpu':
            self.device_manager.set_device(val)
        elif key == 'forceFp32':
            self.device_manager.set_force_fp32(val)
        # FIXME: This is a very counter-intuitive handling of audio modes...
        # Map "serverAudioSampleRate" to "inputSampleRate" and "outputSampleRate"
        # since server audio can have its sample rate configured.
        # Revert change in case we switched back to client audio mode.
        elif key == 'enableServerAudio':
            if val:
                self.update_settings('inputSampleRate', self.settings.serverAudioSampleRate)
                self.update_settings('outputSampleRate', self.settings.serverAudioSampleRate)
            else:
                self.update_settings('inputSampleRate', 48000)
                self.update_settings('outputSampleRate', 48000)
        elif key == 'serverAudioSampleRate':
            self.update_settings('inputSampleRate', self.settings.serverAudioSampleRate)
            self.update_settings('outputSampleRate', self.settings.serverAudioSampleRate)

        self.serverDevice.update_settings(key, val, old_value)
        if self.voiceChanger is not None:
            self.voiceChanger.update_settings(key, val, old_value)

        return self.get_info()

    def changeVoice(self, receivedData: AudioInOut):
        if self.settings.passThrough:  # パススルー
            return receivedData, [0, 0, 0]

        if self.voiceChanger is None:
            logger.info("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1, dtype=np.float32), [0, 0, 0]

        try:
            return self.voiceChanger.on_request(receivedData)
        except NoModeLoadedException as e:
            logger.warn(f"[Voice Changer] [Exception], {e}")
            return np.zeros(1, dtype=np.float32), [0, 0, 0]
        except VoiceChangerIsNotSelectedException:
            logger.warn("[Voice Changer] Voice Changer is not selected. Wait a bit and if there is no improvement, please re-select vc.")
            return np.zeros(1, dtype=np.float32), [0, 0, 0]
        except PipelineNotInitializedException:
            logger.warn("[Voice Changer] Pipeline is not initialized.")
            return np.zeros(1, dtype=np.float32), [0, 0, 0]
        except Exception as e:
            logger.warn(f"[Voice Changer] VC PROCESSING EXCEPTION!!! {e}")
            logger.exception(e)
            return np.zeros(1, dtype=np.float32), [0, 0, 0]

    def export2onnx(self):
        return self.voiceChanger.export2onnx()

    async def merge_models(self, request: str):
        # self.voiceChanger.merge_models(request)
        req = json.loads(request)
        req = ModelMergerRequest(**req)
        req.files = [MergeElement(**f) for f in req.files]
        # Slots range is 0-499
        slot = len(self.modelSlotManager.getAllSlotInfo()) - 1
        if req.voiceChangerType == "RVC":
            merged = RVCModelMerger.merge_models(self.params, req, slot)
            loadParam = LoadModelParams(voiceChangerType="RVC", slot=slot, isSampleMode=False, sampleId="", files=[LoadModelParamFile(name=os.path.basename(merged), kind="rvcModel", dir="")], params={})
            await self.load_model(loadParam)
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
