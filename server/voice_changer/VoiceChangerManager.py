import numpy as np
from voice_changer.VoiceChanger import VoiceChanger
from const import ModelType
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from dataclasses import dataclass, asdict
import torch


@dataclass()
class GPUInfo:
    id: int
    name: str
    memory: int


@dataclass()
class VoiceChangerManagerSettings:
    dummy: int

    # intData: list[str] = field(default_factory=lambda: ["slotIndex"])


class VoiceChangerManager(object):
    _instance = None

    def __init__(self, params: VoiceChangerParams):
        self.voiceChanger: VoiceChanger = None
        self.settings: VoiceChangerManagerSettings = VoiceChangerManagerSettings(dummy=0)
        # スタティックな情報を収集
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
            cls._instance.voiceChanger = VoiceChanger(params)
        return cls._instance

    def loadModel(self, props: LoadModelParams):
        info = self.voiceChanger.loadModel(props)
        if hasattr(info, "status") and info["status"] == "NG":
            return info
        else:
            info["status"] = "OK"
            return info

    def get_info(self):
        data = asdict(self.settings)
        data["gpus"] = self.gpus

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
        if hasattr(self, "voiceChanger"):
            self.voiceChanger.update_settings(key, val)
        else:
            return {"status": "ERROR", "msg": "no model loaded"}
        return self.get_info()

    def changeVoice(self, receivedData: AudioInOut):
        if hasattr(self, "voiceChanger") is True:
            return self.voiceChanger.on_request(receivedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16), []

    def switchModelType(self, modelType: ModelType):
        return self.voiceChanger.switchModelType(modelType)

    def getModelType(self):
        return self.voiceChanger.getModelType()

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
