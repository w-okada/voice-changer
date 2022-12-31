import socketio

from sio.MMVC_Namespace import MMVC_Namespace
from voice_changer.VoiceChangerManager import VoiceChangerManager

class MMVC_SocketIOServer():
    @classmethod
    def get_instance(cls, voiceChangerManager:VoiceChangerManager):
        if not hasattr(cls, "_instance"):
            sio = socketio.AsyncServer(
                    async_mode='asgi',
                    cors_allowed_origins='*'
                )
            namespace = MMVC_Namespace.get_instance(voiceChangerManager)
            sio.register_namespace(namespace)
            cls._instance = sio
            return cls._instance

        return cls._instance

