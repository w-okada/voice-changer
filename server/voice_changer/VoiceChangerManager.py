import numpy as np
from voice_changer.VoiceChanger import VoiceChanger
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class VoiceChangerManager():
    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def loadModel(self, config, model, onnx_model):
        if hasattr(self, 'voiceChanger') == False:
            self.voiceChanger = VoiceChanger(config)
        info = self.voiceChanger.loadModel(config, model, onnx_model)
        info["status"]="OK"
        return info

    def get_info(self):
        if hasattr(self, 'voiceChanger'):
            info = self.voiceChanger.get_info()
            info["status"]="OK"
            return info
        else:
            return {"status":"ERROR", "msg":"no model loaded"}

    def update_setteings(self, key:str, val:any):
        if hasattr(self, 'voiceChanger'):
            info = self.voiceChanger.update_setteings(key, val)
            info["status"]="OK"
            return info
        else:
            return {"status":"ERROR", "msg":"no model loaded"}

    def changeVoice(self, unpackedData:any):
        if hasattr(self, 'voiceChanger') == True:
            return self.voiceChanger.on_request(unpackedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16)
