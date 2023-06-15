import time

import numpy as np
import librosa
import sounddevice as sd

from voice_changer.Local.AudioDeviceList import ServerAudioDevice
from voice_changer.VoiceChanger import VoiceChanger
from voice_changer.utils.Timer import Timer


class ServerDevice:
    def __init__(self):
        self.voiceChanger: VoiceChanger | None = None
        pass

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        if self.voiceChanger is None:
            print("[Voice Changer] voiceChanger is None")
            return

        try:
            indata = indata * self.voiceChanger.settings.serverInputAudioGain
            with Timer("all_inference_time") as t:
                unpackedData = librosa.to_mono(indata.T) * 32768.0
                out_wav, times = self.voiceChanger.on_request(unpackedData)
                outputChunnels = outdata.shape[1]
                outdata[:] = np.repeat(out_wav, outputChunnels).reshape(-1, outputChunnels) / 32768.0
                outdata[:] = outdata * self.voiceChanger.settings.serverOutputAudioGain
            all_inference_time = t.secs
            performance = [all_inference_time] + times
            if self.voiceChanger.emitTo is not None:
                self.voiceChanger.emitTo(performance)
            self.voiceChanger.settings.performance = [round(x * 1000) for x in performance]
        except Exception as e:
            print("[Voice Changer] ex:", e)

    def getServerAudioDevice(self, audioDeviceList: list[ServerAudioDevice], index: int):
        serverAudioDevice = [x for x in audioDeviceList if x.index == index]
        if len(serverAudioDevice) > 0:
            return serverAudioDevice[0]
        else:
            return None

    def serverLocal(self, _vc: VoiceChanger):
        self.voiceChanger = _vc
        vc = self.voiceChanger

        currentInputDeviceId = -1
        currentModelSamplingRate = -1
        currentOutputDeviceId = -1
        currentInputChunkNum = -1
        while True:
            if vc.settings.serverAudioStated == 0 or vc.settings.serverInputDeviceId == -1 or vc is None:
                vc.settings.inputSampleRate = 48000
                time.sleep(2)
            else:
                sd._terminate()
                sd._initialize()

                sd.default.device[0] = vc.settings.serverInputDeviceId
                currentInputDeviceId = vc.settings.serverInputDeviceId
                sd.default.device[1] = vc.settings.serverOutputDeviceId
                currentOutputDeviceId = vc.settings.serverOutputDeviceId

                currentInputChannelNum = vc.settings.serverAudioInputDevices

                serverInputAudioDevice = self.getServerAudioDevice(vc.settings.serverAudioInputDevices, currentInputDeviceId)
                serverOutputAudioDevice = self.getServerAudioDevice(vc.settings.serverAudioOutputDevices, currentOutputDeviceId)
                print(serverInputAudioDevice, serverOutputAudioDevice)
                if serverInputAudioDevice is None or serverOutputAudioDevice is None:
                    time.sleep(2)
                    print("serverInputAudioDevice or serverOutputAudioDevice is None")
                    continue

                currentInputChannelNum = serverInputAudioDevice.maxInputChannels
                currentOutputChannelNum = serverOutputAudioDevice.maxOutputChannels

                currentInputChunkNum = vc.settings.serverReadChunkSize
                block_frame = currentInputChunkNum * 128

                # sample rate precheck(alsa cannot use 40000?)
                try:
                    currentModelSamplingRate = self.voiceChanger.voiceChangerModel.get_processing_sampling_rate()
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
                    vc.settings.serverInputAudioSampleRate = currentModelSamplingRate
                    vc.settings.inputSampleRate = currentModelSamplingRate
                    print(f"[Voice Changer] sample rate {vc.settings.serverInputAudioSampleRate}")
                except Exception as e:
                    print(
                        "[Voice Changer] ex: fallback to device default samplerate",
                        e,
                    )
                    vc.settings.serverInputAudioSampleRate = serverInputAudioDevice.default_samplerate
                    vc.settings.inputSampleRate = vc.settings.serverInputAudioSampleRate

                # main loop
                try:
                    with sd.Stream(
                        callback=self.audio_callback,
                        blocksize=block_frame,
                        samplerate=vc.settings.serverInputAudioSampleRate,
                        dtype="float32",
                        channels=[currentInputChannelNum, currentOutputChannelNum],
                    ):
                        while vc.settings.serverAudioStated == 1 and currentInputDeviceId == vc.settings.serverInputDeviceId and currentOutputDeviceId == vc.settings.serverOutputDeviceId and currentModelSamplingRate == self.voiceChanger.voiceChangerModel.get_processing_sampling_rate() and currentInputChunkNum == vc.settings.serverReadChunkSize:
                            time.sleep(2)
                            print(
                                "[Voice Changer] server audio",
                                vc.settings.performance,
                            )
                            print(
                                "[Voice Changer] info:",
                                vc.settings.serverAudioStated,
                                currentInputDeviceId,
                                currentOutputDeviceId,
                                vc.settings.serverInputAudioSampleRate,
                                currentInputChunkNum,
                            )

                except Exception as e:
                    print("[Voice Changer] ex:", e)
                    time.sleep(2)
