from settings import ServerSettings

from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp
from restapi.MMVC_Rest import MMVC_Rest

settings = ServerSettings()

voice_changer_manager = VoiceChangerManager(settings)
app_fastapi = MMVC_Rest.get_instance(voice_changer_manager, settings.model_dir, settings.allowed_origins, settings.port)
app_socketio = MMVC_SocketIOApp.get_instance(app_fastapi, voice_changer_manager, settings.allowed_origins, settings.port)