from settings import ServerSettings, resolve_paths

from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp
from restapi.MMVC_Rest import MMVC_Rest

settings = resolve_paths(ServerSettings())

voice_changer_manager = VoiceChangerManager(settings)
fastapi = MMVC_Rest.get_instance(voice_changer_manager, settings.model_dir, settings.allowed_origins, settings.port)
socketio = MMVC_SocketIOApp.get_instance(fastapi, voice_changer_manager, settings.allowed_origins, settings.port)