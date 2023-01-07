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
        if hasattr(self, 'voiceChanger') == True:
            self.voiceChanger.destroy()
        self.voiceChanger = VoiceChanger(config, model, onnx_model)

    def get_info(self):
        if hasattr(self, 'voiceChanger'):
            return self.voiceChanger.get_info()
        else:
            return {"no info":"no info"}

    def set_onnx_provider(self, provider:str):
        if hasattr(self, 'voiceChanger'):
            return self.voiceChanger.set_onnx_provider(provider)
        else:
            return {"error":"no voice changer"}


    def changeVoice(self, gpu, srcId, dstId, timestamp, convertChunkNum, crossFadeLowerValue, crossFadeOffsetRate, crossFadeEndRate, unpackedData):
        if hasattr(self, 'voiceChanger') == True:
            return self.voiceChanger.on_request(gpu, srcId, dstId, timestamp, convertChunkNum, crossFadeLowerValue, crossFadeOffsetRate, crossFadeEndRate, unpackedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16)

    def changeVoice_old(self, gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData):
        if hasattr(self, 'voiceChanger') == True:
            return self.voiceChanger.on_request(gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16)
