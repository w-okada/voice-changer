import numpy as np
import threading
from data.ModelSample import ModelSamples
from data.ModelSlot import ModelSlots, loadSlotInfo
from utils.downloader.SampleDownloader import downloadSample, getSampleInfos
from voice_changer.Local.ServerDevice import ServerDevice
from voice_changer.RVC.ModelSlotGenerator import setSlotAsRVC

from voice_changer.VoiceChanger import VoiceChanger
from const import MAX_SLOT_NUM, VoiceChangerType
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams

from dataclasses import dataclass, asdict, field
import torch
import json


@dataclass()
class GPUInfo:
    id: int
    name: str
    memory: int


@dataclass()
class VoiceChangerManagerSettings:
    slotIndex: int
    intData: list[str] = field(default_factory=lambda: ["slotIndex"])


class VoiceChangerManager(object):
    _instance = None

    def __init__(self, params: VoiceChangerParams):
        self.voiceChanger: VoiceChanger = None
        self.settings: VoiceChangerManagerSettings = VoiceChangerManagerSettings(slotIndex=0)
        self.params: VoiceChangerParams = params

        self.serverDevice = ServerDevice()

        # スタティックな情報を収集
        self.sampleModels: list[ModelSamples] = getSampleInfos(self.params.sample_mode)
        self.gpus: list[GPUInfo] = self._get_gpuInfos()

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

            gpu_num = torch.cuda.device_count()
            mps_enabled: bool = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            print(f"VoiceChanger Initialized (GPU_NUM:{gpu_num}, mps_enabled:{mps_enabled})")

            cls._instance.voiceChanger = VoiceChanger(params, cls._instance.settings.slotIndex)
            thread = threading.Thread(target=cls._instance.serverDevice.serverLocal, args=(cls._instance.voiceChanger,))
            thread.start()
            cls._instance.voiceChanger.prepareModel()
        return cls._instance

    def loadModel(self, slot: int, voiceChangerType: VoiceChangerType, params: str):
        print(slot, voiceChangerType, params)
        paramDict = json.loads(params)
        if voiceChangerType == "RVC":
            if "sampleId" in paramDict and len(paramDict["sampleId"]) > 0:
                print("[Voice Canger]: Download RVC sample.")
                downloadSample(self.params.sample_mode, paramDict["sampleId"], self.params.model_dir, slot, {"useIndex": paramDict["rvcIndexDownload"]})
            else:
                print("[Voice Canger]: Set uploaded RVC model to slot.")
                setSlotAsRVC(self.params.model_dir, slot, paramDict)

        return self.get_info()

    def get_slotInfos(self):
        slotInfos: list[ModelSlots] = []
        for slotIndex in range(MAX_SLOT_NUM):
            slotInfo = loadSlotInfo(self.params.model_dir, slotIndex)
            slotInfos.append(slotInfo)
        return slotInfos

    def get_info(self):
        data = asdict(self.settings)
        slotInfos = self.get_slotInfos()
        data["slotInfos"] = slotInfos
        data["gpus"] = self.gpus
        data["sampleModels"] = self.sampleModels

        data["status"] = "OK"
        if hasattr(self, "voiceChanger"):
            info = self.voiceChanger.get_info()
            data.update(info)
            return data
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

    def get_performance(self):
        if hasattr(self, "voiceChanger"):
            info = self.voiceChanger.get_performance()
            return info
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

    def update_settings(self, key: str, val: str | int | float):
        if key in self.settings.intData:
            setattr(self.settings, key, int(val))
            if key == "slotIndex":
                val = val % 1000  # Quick hack for same slot is selected
                setattr(self.settings, key, int(val))

                newVoiceChanger = VoiceChanger(self.params, self.settings.slotIndex)
                newVoiceChanger.prepareModel()
                self.serverDevice.serverLocal(newVoiceChanger)
                del self.voiceChanger
                self.voiceChanger = newVoiceChanger
        elif hasattr(self, "voiceChanger"):
            self.voiceChanger.update_settings(key, val)
        else:
            print(f"[Voice Changer] update is not handled. ({key}:{val})")
        return self.get_info()

    def changeVoice(self, receivedData: AudioInOut):
        if hasattr(self, "voiceChanger") is True:
            return self.voiceChanger.on_request(receivedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16), []

    def export2onnx(self):
        return self.voiceChanger.export2onnx()

    def merge_models(self, request: str):
        return self.voiceChanger.merge_models(request)

    def update_model_default(self):
        return self.voiceChanger.update_model_default()

    def update_model_info(self, newData: str):
        return self.voiceChanger.update_model_info(newData)

    def upload_model_assets(self, params: str):
        return self.voiceChanger.upload_model_assets(params)
