import numpy as np
from voice_changer.VoiceChanger import VoiceChanger
from const import ModelType
from voice_changer.utils.LoadModelParams import LoadModelParams
from voice_changer.utils.VoiceChangerModel import AudioInOut
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class VoiceChangerManager(object):
    _instance = None
    voiceChanger: VoiceChanger = None

    @classmethod
    def get_instance(cls, params: VoiceChangerParams):
        if cls._instance is None:
            cls._instance = cls()
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
        if hasattr(self, "voiceChanger"):
            info = self.voiceChanger.get_info()
            info["status"] = "OK"
            return info
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

    def update_settings(self, key: str, val: str | int | float):
        if hasattr(self, "voiceChanger"):
            info = self.voiceChanger.update_settings(key, val)
            info["status"] = "OK"
            return info
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

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
