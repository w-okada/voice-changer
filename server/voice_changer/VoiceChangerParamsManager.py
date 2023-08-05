from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class VoiceChangerParamsManager:
    _instance = None

    def __init__(self):
        self.params = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def setParams(self, params: VoiceChangerParams):
        self.params = params
