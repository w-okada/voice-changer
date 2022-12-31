import numpy as np
from mods.VoiceChanger import VoiceChanger

class VoiceChangerManager():
    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def loadModel(self, config, model):
        if hasattr(self, 'voiceChanger') == True:
            self.voiceChanger.destroy()
        self.voiceChanger = VoiceChanger(config, model)

    def changeVoice(self, gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData):
        if hasattr(self, 'voiceChanger') == True:
            return self.voiceChanger.on_request(gpu, srcId, dstId, timestamp, prefixChunkSize, unpackedData)
        else:
            print("Voice Change is not loaded. Did you load a correct model?")
            return np.zeros(1).astype(np.int16)
