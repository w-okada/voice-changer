import sounddevice as sd
from dataclasses import dataclass
from const import ServerAudioDeviceTypes


@dataclass
class ServerAudioDevice:
    kind: ServerAudioDeviceTypes = ServerAudioDeviceTypes.audioinput
    index: int = 0
    name: str = ""
    hostAPI: str = ""
    maxInputChannels: int = 0
    maxOutputChannels: int = 0


def list_audio_device():
    audioDeviceList = sd.query_devices()

    inputAudioDeviceList = [d for d in audioDeviceList if d["max_input_channels"] > 0]
    outputDeviceList = [d for d in audioDeviceList if d["max_output_channels"] > 0]
    hostapis = sd.query_hostapis()

    print("input:", inputAudioDeviceList)
    print("output:", outputDeviceList)
    print("hostapis", hostapis)

    serverAudioInputDevices = []
    serverAudioOutputDevices = []
    for d in inputAudioDeviceList:
        serverInputAudioDevice: ServerAudioDevice = ServerAudioDevice(
            kind=ServerAudioDeviceTypes.audioinput,
            index=d["index"],
            name=d["name"],
            hostAPI=hostapis[d["hostapi"]]["name"],
            maxInputChannels=d["max_input_channels"],
            maxOutputChannels=d["max_output_channels"],
        )
        serverAudioInputDevices.append(serverInputAudioDevice)
    for d in outputDeviceList:
        serverOutputAudioDevice: ServerAudioDevice = ServerAudioDevice(
            kind=ServerAudioDeviceTypes.audiooutput,
            index=d["index"],
            name=d["name"],
            hostAPI=hostapis[d["hostapi"]]["name"],
            maxInputChannels=d["max_input_channels"],
            maxOutputChannels=d["max_output_channels"],
        )
        serverAudioOutputDevices.append(serverOutputAudioDevice)

    return serverAudioInputDevices, serverAudioOutputDevices
