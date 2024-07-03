import os
from const import ROOT_PATH
# NOTE: This is required to fix current working directory on macOS
os.chdir(ROOT_PATH)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from voice_changer.VoiceChangerManager import VoiceChangerManager
from sio.MMVC_SocketIOApp import MMVC_SocketIOApp
from restapi.MMVC_Rest import MMVC_Rest
from settings import ServerSettings

settings = ServerSettings()

voice_changer_manager = VoiceChangerManager(settings)
fastapi = MMVC_Rest.get_instance(voice_changer_manager, settings.model_dir, settings.allowed_origins, settings.port)
socketio = MMVC_SocketIOApp.get_instance(fastapi, voice_changer_manager, settings.allowed_origins, settings.port)