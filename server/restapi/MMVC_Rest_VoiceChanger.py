import base64
import struct
import numpy as np
import traceback
import pyaudio

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from voice_changer.VoiceChangerManager import VoiceChangerManager
from pydantic import BaseModel
import threading


class VoiceModel(BaseModel):
    timestamp: int
    buffer: str


class MMVC_Rest_VoiceChanger:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.voiceChangerManager = voiceChangerManager
        self.router = APIRouter()
        self.router.add_api_route("/test", self.test, methods=["POST"])
        self.router.add_api_route("/microphone", self.get_microphone, methods=["GET"])

        self.tlock = threading.Lock()

    def get_microphone(self):
        audio = pyaudio.PyAudio()
        audio_input_devices = []
        audio_output_devices = []
        audio_devices = {}
        host_apis = []

        for api_index in range(audio.get_host_api_count()):
            host_apis.append(audio.get_host_api_info_by_index(api_index)['name'])

        for x in range(0, audio.get_device_count()):
            device = audio.get_device_info_by_index(x)
            try:
                deviceName = device['name'].encode('shift-jis').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                deviceName = device['name']

            deviceIndex = device['index']
            hostAPI = host_apis[device['hostApi']]

            if device['maxInputChannels'] > 0:
                audio_input_devices.append({"kind": "audioinput", "index": deviceIndex, "name": deviceName, "hostAPI": hostAPI})
            if device['maxOutputChannels'] > 0:
                audio_output_devices.append({"kind": "audiooutput", "index": deviceIndex, "name": deviceName, "hostAPI": hostAPI})
        audio_devices["audio_input_devices"] = audio_input_devices
        audio_devices["audio_output_devices"] = audio_output_devices

        json_compatible_item_data = jsonable_encoder(audio_devices)
        return JSONResponse(content=json_compatible_item_data)

    def test(self, voice: VoiceModel):
        try:
            timestamp = voice.timestamp
            buffer = voice.buffer
            wav = base64.b64decode(buffer)

            if wav == 0:
                samplerate, data = read("dummy.wav")
                unpackedData = data
            else:
                unpackedData = np.array(struct.unpack(
                    '<%sh' % (len(wav) // struct.calcsize('<h')), wav))
                # write("logs/received_data.wav", 24000,
                #       unpackedData.astype(np.int16))

            self.tlock.acquire()
            changedVoice = self.voiceChangerManager.changeVoice(unpackedData)
            self.tlock.release()

            changedVoiceBase64 = base64.b64encode(changedVoice).decode('utf-8')
            data = {
                "timestamp": timestamp,
                "changedVoiceBase64": changedVoiceBase64
            }

            json_compatible_item_data = jsonable_encoder(data)
            return JSONResponse(content=json_compatible_item_data)

        except Exception as e:
            print("REQUEST PROCESSING!!!! EXCEPTION!!!", e)
            print(traceback.format_exc())
            self.tlock.release()
            return str(e)
