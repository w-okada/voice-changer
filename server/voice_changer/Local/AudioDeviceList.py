import sounddevice as sd
from dataclasses import dataclass, field

import numpy as np

from const import ServerAudioDeviceType
from mods.log_control import VoiceChangaerLogger
# from const import SERVER_DEVICE_SAMPLE_RATES

logger = VoiceChangaerLogger.get_instance().getLogger()


@dataclass
class ServerAudioDevice:
    kind: ServerAudioDeviceType = "audioinput"
    index: int = 0
    name: str = ""
    hostAPI: str = ""
    maxInputChannels: int = 0
    maxOutputChannels: int = 0
    default_samplerate: int = 0
    available_samplerates: list[int] = field(default_factory=lambda: [])


def dummy_callback(data: np.ndarray, frames, times, status):
    pass


def checkSamplingRate(deviceId: int, desiredSamplingRate: int, type: ServerAudioDeviceType):
    if type == "input":
        try:
            with sd.InputStream(
                device=deviceId,
                callback=dummy_callback,
                dtype="float32",
                samplerate=desiredSamplingRate
            ):
                pass
            return True
        except Exception as e: # NOQA
            # print("[checkSamplingRate]", e)
            return False
    else:
        try:
            with sd.OutputStream(
                device=deviceId,
                callback=dummy_callback,
                dtype="float32",
                samplerate=desiredSamplingRate
            ):
                pass
            return True
        except Exception as e: # NOQA
            # print("[checkSamplingRate]", e)
            return False


def list_audio_device():
    try:
        audioDeviceList = sd.query_devices()
    except Exception as e:
        logger.error("[Voice Changer] ex:query_devices")
        logger.exception(e)
        raise e

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
            kind="audioinput",
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
            kind="audiooutput",
            index=d["index"],
            name=d["name"],
            hostAPI=hostapis[d["hostapi"]]["name"],
            maxInputChannels=d["max_input_channels"],
            maxOutputChannels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        serverAudioOutputDevices.append(serverOutputAudioDevice)

    # print("check sample rate1")
    # for d in serverAudioInputDevices:
    #     print("check sample rate1-1")
    #     for sr in SERVER_DEVICE_SAMPLE_RATES:
    #         print("check sample rate1-2")
    #         if checkSamplingRate(d.index, sr, "input"):
    #             d.available_samplerates.append(sr)
    # print("check sample rate2")
    # for d in serverAudioOutputDevices:
    #     print("check sample rate2-1")
    #     for sr in SERVER_DEVICE_SAMPLE_RATES:
    #         print("check sample rate2-2")
    #         if checkSamplingRate(d.index, sr, "output"):
    #             d.available_samplerates.append(sr)
    # print("check sample rate3")

    return serverAudioInputDevices, serverAudioOutputDevices
