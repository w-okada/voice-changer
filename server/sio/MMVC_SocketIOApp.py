import socketio
from mods.log_control import VoiceChangaerLogger

from sio.MMVC_SocketIOServer import MMVC_SocketIOServer
from voice_changer.VoiceChangerManager import VoiceChangerManager
from const import getFrontendPath

logger = VoiceChangaerLogger.get_instance().getLogger()


class MMVC_SocketIOApp:
    _instance: socketio.ASGIApp | None = None

    @classmethod
    def get_instance(cls, app_fastapi, voiceChangerManager: VoiceChangerManager):
        if cls._instance is None:
            logger.info("[Voice Changer] MMVC_SocketIOApp initializing...")
            sio = MMVC_SocketIOServer.get_instance(voiceChangerManager)
            app_socketio = socketio.ASGIApp(
                sio,
                other_asgi_app=app_fastapi,
                static_files={
                    "/assets/icons/github.svg": {
                        "filename": f"{getFrontendPath()}/assets/icons/github.svg",
                        "content_type": "image/svg+xml",
                    },
                    "/assets/icons/help-circle.svg": {
                        "filename": f"{getFrontendPath()}/assets/icons/help-circle.svg",
                        "content_type": "image/svg+xml",
                    },
                    "/assets/icons/tool.svg": {
                        "filename": f"{getFrontendPath()}/assets/icons/tool.svg",
                        "content_type": "image/svg+xml",
                    },
                    "/assets/icons/folder.svg": {
                        "filename": f"{getFrontendPath()}/assets/icons/folder.svg",
                        "content_type": "image/svg+xml",
                    },
                    "/buymeacoffee.png": {
                        "filename": f"{getFrontendPath()}/assets/buymeacoffee.png",
                        "content_type": "image/png",
                    },
                    "": f"{getFrontendPath()}",
                    "/": f"{getFrontendPath()}/index.html",
                },
            )

            cls._instance = app_socketio
            logger.info("[Voice Changer] MMVC_SocketIOApp initializing... done.")
            return cls._instance

        return cls._instance
