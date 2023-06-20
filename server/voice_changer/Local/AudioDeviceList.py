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
    default_samplerate: int = 0


def list_audio_device():
    try:
        audioDeviceList = sd.query_devices()
    except Exception as e:
        print("[Voice Changer] ex:query_devices")
        print(e)
        return [], []

    inputAudioDeviceList = [d for d in audioDeviceList if d["max_input_channels"] > 0]
    outputAudioDeviceList = [d for d in audioDeviceList if d["max_output_channels"] > 0]
    hostapis = sd.query_hostapis()

    # print("input:", inputAudioDeviceList)
    # print("output:", outputDeviceList)
    # print("hostapis", hostapis)

    serverAudioInputDevices: list[ServerAudioDevice] = []
    serverAudioOutputDevices: list[ServerAudioDevice] = []
    for d in inputAudioDeviceList:
        serverInputAudioDevice: ServerAudioDevice = ServerAudioDevice(
            kind=ServerAudioDeviceTypes.audioinput,
            index=d["index"],
            name=d["name"],
            hostAPI=hostapis[d["hostapi"]]["name"],
            maxInputChannels=d["max_input_channels"],
            maxOutputChannels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        serverAudioInputDevices.append(serverInputAudioDevice)
    for d in outputAudioDeviceList:
        serverOutputAudioDevice: ServerAudioDevice = ServerAudioDevice(
            kind=ServerAudioDeviceTypes.audiooutput,
            index=d["index"],
            name=d["name"],
            hostAPI=hostapis[d["hostapi"]]["name"],
            maxInputChannels=d["max_input_channels"],
            maxOutputChannels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        serverAudioOutputDevices.append(serverOutputAudioDevice)

    return serverAudioInputDevices, serverAudioOutputDevices
