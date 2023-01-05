import socketio

from sio.MMVC_SocketIOServer import MMVC_SocketIOServer
from voice_changer.VoiceChangerManager import VoiceChangerManager
from const import frontend_path

class MMVC_SocketIOApp():
    @classmethod
    def get_instance(cls, app_fastapi, voiceChangerManager:VoiceChangerManager):
        print("INDEX:::", f'${frontend_path}/index.html')
        if not hasattr(cls, "_instance"):
            sio = MMVC_SocketIOServer.get_instance(voiceChangerManager)
            app_socketio = socketio.ASGIApp(
                    sio,
                    other_asgi_app=app_fastapi,
                    static_files={
                        '/assets/icons/github.svg': {
                            'filename': f'{frontend_path}/assets/icons/github.svg',
                            'content_type': 'image/svg+xml'
                        },
                        '': f'{frontend_path}',
                        '/': f'{frontend_path}/index.html',
                    }
                )


            cls._instance = app_socketio
            return cls._instance

        return cls._instance

