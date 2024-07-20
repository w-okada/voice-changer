import socketio

from typing import Literal
from sio.MMVC_Namespace import MMVC_Namespace
from voice_changer.VoiceChangerManager import VoiceChangerManager


class MMVC_SocketIOServer:
    _instance: socketio.AsyncServer | None = None

    @classmethod
    def get_instance(
        cls,
        voiceChangerManager: VoiceChangerManager,
        allowedOrigins: list[str] | Literal['*'],
    ):
        if cls._instance is None:
            sio = socketio.AsyncServer(async_mode="asgi", serializer='msgpack', cors_allowed_origins=allowedOrigins)
            namespace = MMVC_Namespace.get_instance(voiceChangerManager)
            sio.register_namespace(namespace)
            cls._instance = sio
            return cls._instance

        return cls._instance
