from dataclasses import dataclass, asdict

import numpy as np

from voice_changer.Local.AudioDeviceList import list_audio_device
import time
import sounddevice as sd
from voice_changer.utils.Timer import Timer
import librosa

from voice_changer.utils.VoiceChangerModel import AudioInOut
from typing import Protocol


@dataclass
class ServerDeviceSettings:
    enableServerAudio: int = 0  # 0:off, 1:on
    serverAudioStated: int = 0  # 0:off, 1:on
    serverInputAudioSampleRate: int = 44100
    serverOutputAudioSampleRate: int = 44100
    serverInputDeviceId: int = -1
    serverOutputDeviceId: int = -1
    serverReadChunkSize: int = 256
    serverInputAudioGain: float = 1.0
    serverOutputAudioGain: float = 1.0


EditableServerDeviceSettings = {
    "intData": [
        "enableServerAudio",
        "serverAudioStated",
        "serverInputAudioSampleRate",
        "serverOutputAudioSampleRate",
        "serverInputDeviceId",
        "serverOutputDeviceId",
        "serverReadChunkSize",
    ],
    "floatData": [
        "serverInputAudioGain",
        "serverOutputAudioGain",
    ],
}


class ServerDeviceCallbacks(Protocol):
    def on_request(self, unpackedData: AudioInOut):
        ...

    def emitTo(self, performance: list[float]):
        ...

    def get_processing_sampling_rate(self):
        ...

    def setSamplingRate(self, sr: int):
        ...


class ServerDevice:
    def __init__(self, serverDeviceCallbacks: ServerDeviceCallbacks):
        self.settings = ServerDeviceSettings()
        self.serverDeviceCallbacks = serverDeviceCallbacks

    def getServerInputAudioDevice(self, index: int):
        audioinput, _audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audioinput if x.index == index]
        if len(serverAudioDevice) > 0:
            return serverAudioDevice[0]
        else:
            return None

    def getServerOutputAudioDevice(self, index: int):
        _audioinput, audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audiooutput if x.index == index]
        if len(serverAudioDevice) > 0:
            return serverAudioDevice[0]
        else:
            return None

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        try:
            indata = indata * self.settings.serverInputAudioGain
            with Timer("all_inference_time") as t:
                unpackedData = librosa.to_mono(indata.T) * 32768.0
                out_wav, times = self.serverDeviceCallbacks.on_request(unpackedData)
                outputChunnels = outdata.shape[1]
                outdata[:] = np.repeat(out_wav, outputChunnels).reshape(-1, outputChunnels) / 32768.0
                outdata[:] = outdata * self.settings.serverOutputAudioGain
            all_inference_time = t.secs
            self.performance = [all_inference_time] + times
            self.serverDeviceCallbacks.emitTo(self.performance)
            self.performance = [round(x * 1000) for x in self.performance]
        except Exception as e:
            print("[Voice Changer] ex:", e)

    def start(self):
        # currentInputDeviceId = -1
        # currentOutputDeviceId = -1
        # currentInputChunkNum = -1
        currentModelSamplingRate = -1
        while True:
            if self.settings.serverAudioStated == 0 or self.settings.serverInputDeviceId == -1:
                # self.settings.inputSampleRate = 48000
                time.sleep(2)
            else:
                sd._terminate()
                sd._initialize()

                sd.default.device[0] = self.settings.serverInputDeviceId
                sd.default.device[1] = self.settings.serverOutputDeviceId

                serverInputAudioDevice = self.getServerInputAudioDevice(sd.default.device[0])
                serverOutputAudioDevice = self.getServerOutputAudioDevice(sd.default.device[1])
                print("Devices:", serverInputAudioDevice, serverOutputAudioDevice)
                if serverInputAudioDevice is None or serverOutputAudioDevice is None:
                    time.sleep(2)
                    print("serverInputAudioDevice or serverOutputAudioDevice is None")
                    continue

                sd.default.channels[0] = serverInputAudioDevice.maxInputChannels
                sd.default.channels[1] = serverOutputAudioDevice.maxOutputChannels

                currentInputChunkNum = self.settings.serverReadChunkSize
                block_frame = currentInputChunkNum * 128

                # sample rate precheck(alsa cannot use 40000?)
                try:
                    currentModelSamplingRate = self.serverDeviceCallbacks.get_processing_sampling_rate()
                except Exception as e:
                    print("[Voice Changer] ex: get_processing_sampling_rate", e)
                    continue
                try:
                    with sd.Stream(
                        callback=self.audio_callback,
                        blocksize=block_frame,
                        # samplerate=currentModelSamplingRate,
                        dtype="float32",
                        # channels=[currentInputChannelNum, currentOutputChannelNum],
                    ):
                        pass
                    self.settings.serverInputAudioSampleRate = currentModelSamplingRate
                    self.serverDeviceCallbacks.setSamplingRate(currentModelSamplingRate)
                    print(f"[Voice Changer] sample rate {self.settings.serverInputAudioSampleRate}")
                except Exception as e:
                    print("[Voice Changer] ex: fallback to device default samplerate", e)
                    print("[Voice Changer] device default samplerate", serverInputAudioDevice.default_samplerate)
                    self.settings.serverInputAudioSampleRate = round(serverInputAudioDevice.default_samplerate)
                    self.serverDeviceCallbacks.setSamplingRate(round(serverInputAudioDevice.default_samplerate))

                sd.default.samplerate = self.settings.serverInputAudioSampleRate
                sd.default.blocksize = block_frame
                # main loop
                try:
                    with sd.Stream(
                        callback=self.audio_callback,
                        # blocksize=block_frame,
                        # samplerate=vc.settings.serverInputAudioSampleRate,
                        dtype="float32",
                        # channels=[currentInputChannelNum, currentOutputChannelNum],
                    ):
                        while self.settings.serverAudioStated == 1 and sd.default.device[0] == self.settings.serverInputDeviceId and sd.default.device[1] == self.settings.serverOutputDeviceId and currentModelSamplingRate == self.serverDeviceCallbacks.get_processing_sampling_rate() and currentInputChunkNum == self.settings.serverReadChunkSize:
                            time.sleep(2)
                            print("[Voice Changer] server audio", self.performance)
                            print(f"[Voice Changer] started:{self.settings.serverAudioStated}, input:{sd.default.device[0]}, output:{sd.default.device[1]}, mic_sr:{self.settings.serverInputAudioSampleRate}, model_sr:{currentModelSamplingRate}, chunk:{currentInputChunkNum}, ch:[{sd.default.channels}]")

                except Exception as e:
                    print("[Voice Changer] ex:", e)
                    time.sleep(2)

    def get_info(self):
        data = asdict(self.settings)
        audioinput, audiooutput = list_audio_device()
        data["serverAudioInputDevices"] = audioinput
        data["serverAudioOutputDevices"] = audiooutput

        return data

    def update_settings(self, key: str, val: str | int | float):
        if key in EditableServerDeviceSettings["intData"]:
            setattr(self.settings, key, int(val))
        elif key in EditableServerDeviceSettings["floatData"]:
            setattr(self.settings, key, float(val))
        return self.get_info()
