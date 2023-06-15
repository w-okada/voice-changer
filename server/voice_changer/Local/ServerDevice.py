import time

import numpy as np
import librosa
import sounddevice as sd

from dataclasses import dataclass, asdict, field

from voice_changer.Local.AudioDeviceList import ServerAudioDevice
from voice_changer.VoiceChangerManager import VoiceChangerManager
from voice_changer.utils.Timer import Timer


@dataclass()
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


class ServerDevice:
    def __init__(self, voiceChangerManager: VoiceChangerManager):
        self.settings = ServerDeviceSettings()
        self.voiceChangerManager: VoiceChangerManager = voiceChangerManager

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        if self.voiceChangerManager.voiceChanger is None:
            print("[Voice Changer] voiceChanger is None")
            return

        try:
            indata = indata * self.settings.serverInputAudioGain
            with Timer("all_inference_time") as t:
                unpackedData = librosa.to_mono(indata.T) * 32768.0
                out_wav, times = self.voiceChangerManager.voiceChanger.on_request(unpackedData)
                outputChunnels = outdata.shape[1]
                outdata[:] = np.repeat(out_wav, outputChunnels).reshape(-1, outputChunnels) / 32768.0
                outdata[:] = outdata * self.settings.serverOutputAudioGain
            all_inference_time = t.secs
            performance = [all_inference_time] + times
            if self.voiceChangerManager.voiceChanger.emitTo is not None:
                self.voiceChangerManager.voiceChanger.emitTo(performance)
            self.voiceChangerManager.voiceChanger.settings.performance = [round(x * 1000) for x in performance]
        except Exception as e:
            print("[Voice Changer] ex:", e)

    def getServerAudioDevice(self, audioDeviceList: list[ServerAudioDevice], index: int):
        serverAudioDevice = [x for x in audioDeviceList if x.index == index]
        if len(serverAudioDevice) > 0:
            return serverAudioDevice[0]
        else:
            return None

    def serverLocal(self):
        currentInputDeviceId = -1
        currentModelSamplingRate = -1
        currentOutputDeviceId = -1
        currentInputChunkNum = -1
        while True:
            if self.settings.serverAudioStated == 0 or self.settings.serverInputDeviceId == -1 or self.voiceChangerManager is None:
                self.voiceChangerManager.voiceChanger.settings.inputSampleRate = 48000
                time.sleep(2)
            else:
                sd._terminate()
                sd._initialize()

                sd.default.device[0] = self.settings.serverInputDeviceId
                currentInputDeviceId = self.settings.serverInputDeviceId
                sd.default.device[1] = self.settings.serverOutputDeviceId
                currentOutputDeviceId = self.settings.serverOutputDeviceId

                serverInputAudioDevice = self.getServerAudioDevice(self.voiceChangerManager.serverAudioInputDevices, currentInputDeviceId)
                serverOutputAudioDevice = self.getServerAudioDevice(self.voiceChangerManager.serverAudioOutputDevices, currentOutputDeviceId)
                print(serverInputAudioDevice, serverOutputAudioDevice)
                if serverInputAudioDevice is None or serverOutputAudioDevice is None:
                    time.sleep(2)
                    print("serverInputAudioDevice or serverOutputAudioDevice is None")
                    continue

                currentInputChannelNum = serverInputAudioDevice.maxInputChannels
                currentOutputChannelNum = serverOutputAudioDevice.maxOutputChannels

                currentInputChunkNum = self.settings.serverReadChunkSize
                block_frame = currentInputChunkNum * 128

                # sample rate precheck(alsa cannot use 40000?)
                try:
                    currentModelSamplingRate = self.voiceChangerManager.voiceChanger.voiceChangerModel.get_processing_sampling_rate()
                except Exception as e:
                    print("[Voice Changer] ex: get_processing_sampling_rate", e)
                    continue
                try:
                    with sd.Stream(
                        callback=self.audio_callback,
                        blocksize=block_frame,
                        samplerate=currentModelSamplingRate,
                        dtype="float32",
                        channels=[currentInputChannelNum, currentOutputChannelNum],
                    ):
                        pass
                    self.settings.serverInputAudioSampleRate = currentModelSamplingRate
                    self.voiceChangerManager.voiceChanger.settings.inputSampleRate = currentModelSamplingRate
                    print(f"[Voice Changer] sample rate {self.settings.serverInputAudioSampleRate}")
                except Exception as e:
                    print(
                        "[Voice Changer] ex: fallback to device default samplerate",
                        e,
                    )
                    self.settings.serverInputAudioSampleRate = serverInputAudioDevice.default_samplerate
                    self.voiceChangerManager.voiceChanger.settings.inputSampleRate = self.settings.serverInputAudioSampleRate

                # main loop
                try:
                    with sd.Stream(
                        callback=self.audio_callback,
                        blocksize=block_frame,
                        samplerate=self.settings.serverInputAudioSampleRate,
                        dtype="float32",
                        channels=[currentInputChannelNum, currentOutputChannelNum],
                    ):
                        while self.settings.serverAudioStated == 1 and currentInputDeviceId == self.settings.serverInputDeviceId and currentOutputDeviceId == self.settings.serverOutputDeviceId and currentModelSamplingRate == self.voiceChangerManager.voiceChanger.voiceChangerModel.get_processing_sampling_rate() and currentInputChunkNum == self.settings.serverReadChunkSize:
                            time.sleep(2)
                            print(
                                "[Voice Changer] server audio",
                                self.voiceChangerManager.settings.performance,
                            )
                            print(
                                "[Voice Changer] info:",
                                self.settings.serverAudioStated,
                                currentInputDeviceId,
                                currentOutputDeviceId,
                                self.settings.serverInputAudioSampleRate,
                                currentInputChunkNum,
                            )

                except Exception as e:
                    print("[Voice Changer] ex:", e)
                    time.sleep(2)
