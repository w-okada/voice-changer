from settings import ServerSettings


class VoiceChangerParamsManager:
    _instance = None

    def __init__(self):
        self.params = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def setParams(self, params: ServerSettings):
        self.params = params

    def getParams(self):
        return self.params
