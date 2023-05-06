import pyaudio

# import json


from dataclasses import dataclass

from const import ServerAudioDeviceTypes


@dataclass
class ServerAudioDevice:
    kind: ServerAudioDeviceTypes = ServerAudioDeviceTypes.audioinput
    index: int = 0
    name: str = ""
    hostAPI: str = ""


def list_audio_device():
    audio = pyaudio.PyAudio()
    audio_input_devices: list[ServerAudioDevice] = []
    audio_output_devices: list[ServerAudioDevice] = []
    # audio_devices = {}
    host_apis = []

    for api_index in range(audio.get_host_api_count()):
        host_apis.append(audio.get_host_api_info_by_index(api_index)["name"])

    for x in range(0, audio.get_device_count()):
        device = audio.get_device_info_by_index(x)
        try:
            deviceName = device["name"].encode("shift-jis").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            deviceName = device["name"]

        deviceIndex = device["index"]
        hostAPI = host_apis[device["hostApi"]]

        if device["maxInputChannels"] > 0:
            audio_input_devices.append(
                ServerAudioDevice(
                    kind=ServerAudioDeviceTypes.audioinput,
                    index=deviceIndex,
                    name=deviceName,
                    hostAPI=hostAPI,
                )
            )
        if device["maxOutputChannels"] > 0:
            audio_output_devices.append(
                ServerAudioDevice(
                    kind=ServerAudioDeviceTypes.audiooutput,
                    index=deviceIndex,
                    name=deviceName,
                    hostAPI=hostAPI,
                )
            )

    return audio_input_devices, audio_output_devices
