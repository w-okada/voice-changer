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
        self.voiceChanger.loadModel(config, model, onnx_model)

    def get_info(self):
        if hasattr(self, 'voiceChanger'):
            return self.voiceChanger.get_info()
        else:
            return {"no info":"no info"}

    def update_setteings(self, key:str, val:any):
        if hasattr(self, 'voiceChanger'):
            return self.voiceChanger.update_setteings(key, val)
        else:
            return {"no info":"no info"}

    def changeVoice(self, unpackedData:any):
        if hasattr(self, 'voiceChanger') == True:
            return self.voiceChanger.on_request(unpackedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16)
