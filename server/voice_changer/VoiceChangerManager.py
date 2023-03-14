import numpy as np
from voice_changer.VoiceChanger import VoiceChanger


class VoiceChangerManager():
    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
            cls._instance.voiceChanger = VoiceChanger()
        return cls._instance

    def loadModel(self, config, model, onnx_model, clusterTorchModel, hubertTorchModel):
        # !! 注意 !! hubertTorchModelは固定値で上書きされるため、設定しても効果ない。
        info = self.voiceChanger.loadModel(config, model, onnx_model, clusterTorchModel, hubertTorchModel)
        info["status"] = "OK"
        return info

    def get_info(self):
        if hasattr(self, 'voiceChanger'):
            info = self.voiceChanger.get_info()
            info["status"] = "OK"
            return info
        else:
            return {"status": "ERROR", "msg": "no model loaded"}

    def update_setteings(self, key: str, val: any):
        if hasattr(self, 'voiceChanger'):
            info = self.voiceChanger.update_setteings(key, val)
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
