import numpy as np
from voice_changer.VoiceChanger import VoiceChanger
from const import ModelType


class VoiceChangerManager():
    @classmethod
    def get_instance(cls, params):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
            cls._instance.voiceChanger = VoiceChanger(params)
        return cls._instance

    def loadModel(self, config, model, onnx_model, clusterTorchModel, feature_file, index_file, is_half: bool = True):
        info = self.voiceChanger.loadModel(config, model, onnx_model, clusterTorchModel, feature_file, index_file, is_half)
        if hasattr(info, "status") and info["status"] == "NG":
            return info
        else:
            info["status"] = "OK"
            return info

    def get_info(self):
        if hasattr(self, 'voiceChanger'):
            info = self.voiceChanger.get_info()
            info["status"] = "OK"
            return info
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

    def update_settings(self, key: str, val: any):
        if hasattr(self, 'voiceChanger'):
            info = self.voiceChanger.update_settings(key, val)
            info["status"] = "OK"
            return info
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

    def changeVoice(self, receivedData: any):
        if hasattr(self, 'voiceChanger') == True:
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
