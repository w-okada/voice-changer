from dataclasses import dataclass, asdict

import numpy as np
from const import SERVER_DEVICE_SAMPLE_RATES

from queue import Queue

from voice_changer.Local.AudioDeviceList import checkSamplingRate, list_audio_device
import time
import sounddevice as sd
from voice_changer.utils.Timer import Timer
import librosa

from voice_changer.utils.VoiceChangerModel import AudioInOut
from typing import Protocol
from typing import Union
from typing import Literal, TypeAlias
AudioDeviceKind: TypeAlias = Literal["input", "output"]


@dataclass
class ServerDeviceSettings:
    enableServerAudio: int = 0  # 0:off, 1:on
    serverAudioStated: int = 0  # 0:off, 1:on
    serverInputAudioSampleRate: int = 44100
    serverOutputAudioSampleRate: int = 44100
    serverMonitorAudioSampleRate: int = 44100

    serverAudioSampleRate: int = 44100
    # serverAudioSampleRate: int = 16000
    # serverAudioSampleRate: int = 48000

    serverInputDeviceId: int = -1
    serverOutputDeviceId: int = -1
    serverMonitorDeviceId: int = -1  # -1 でモニター無効
    serverReadChunkSize: int = 256
    serverInputAudioGain: float = 1.0
    serverOutputAudioGain: float = 1.0

    exclusiveMode: bool = False


EditableServerDeviceSettings = {
    "intData": [
        "enableServerAudio",
        "serverAudioStated",
        "serverInputAudioSampleRate",
        "serverOutputAudioSampleRate",
        "serverMonitorAudioSampleRate",
        "serverAudioSampleRate",
        "serverInputDeviceId",
        "serverOutputDeviceId",
        "serverMonitorDeviceId",
        "serverReadChunkSize",
    ],
    "floatData": [
        "serverInputAudioGain",
        "serverOutputAudioGain",
    ],
    "boolData": [
        "exclusiveMode"
    ]
}


class ServerDeviceCallbacks(Protocol):
    def on_request(self, unpackedData: AudioInOut) -> tuple[AudioInOut, list[Union[int, float]]]:
        ...

    def emitTo(self, performance: list[float]):
        ...

    def get_processing_sampling_rate(self):
        ...

    def setInputSamplingRate(self, sr: int):
        ...

    def setOutputSamplingRate(self, sr: int):
        ...


class ServerDevice:
    def __init__(self, serverDeviceCallbacks: ServerDeviceCallbacks):
        self.settings = ServerDeviceSettings()
        self.serverDeviceCallbacks = serverDeviceCallbacks
        self.out_wav = None
        self.mon_wav = None
        self.serverAudioInputDevices = None
        self.serverAudioOutputDevices = None
        self.outQueue = Queue()
        self.monQueue = Queue()

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
                unpackedData = unpackedData.astype(np.int16)
                out_wav, times = self.serverDeviceCallbacks.on_request(unpackedData)
                outputChannels = outdata.shape[1]
                outdata[:] = np.repeat(out_wav, outputChannels).reshape(-1, outputChannels) / 32768.0
                outdata[:] = outdata * self.settings.serverOutputAudioGain
            all_inference_time = t.secs
            self.performance = [all_inference_time] + times
            self.serverDeviceCallbacks.emitTo(self.performance)
            self.performance = [round(x * 1000) for x in self.performance]
        except Exception as e:
            print("[Voice Changer] ex:", e)

    def audioInput_callback(self, indata: np.ndarray, frames, times, status):
        try:
            indata = indata * self.settings.serverInputAudioGain
            with Timer("all_inference_time") as t:
                unpackedData = librosa.to_mono(indata.T) * 32768.0
                unpackedData = unpackedData.astype(np.int16)
                out_wav, times = self.serverDeviceCallbacks.on_request(unpackedData)
                self.outQueue.put(out_wav)
                self.monQueue.put(out_wav)
            all_inference_time = t.secs
            self.performance = [all_inference_time] + times
            self.serverDeviceCallbacks.emitTo(self.performance)
            self.performance = [round(x * 1000) for x in self.performance]
        except Exception as e:
            print("[Voice Changer][ServerDevice][audioInput_callback] ex:", e)
            # import traceback
            # traceback.print_exc()

    def audioOutput_callback(self, outdata: np.ndarray, frames, times, status):
        try:
            out_wav = self.outQueue.get()
            while self.outQueue.qsize() > 0:
                self.outQueue.get()
            outputChannels = outdata.shape[1]
            outdata[:] = np.repeat(out_wav, outputChannels).reshape(-1, outputChannels) / 32768.0
            outdata[:] = outdata * self.settings.serverOutputAudioGain
        except Exception as e:
            print("[Voice Changer][ServerDevice][audioOutput_callback]  ex:", e)
            # import traceback
            # traceback.print_exc()

    def audioMonitor_callback(self, outdata: np.ndarray, frames, times, status):
        try:
            mon_wav = self.monQueue.get()
            while self.monQueue.qsize() > 0:
                self.monQueue.get()
            outputChannels = outdata.shape[1]
            outdata[:] = np.repeat(mon_wav, outputChannels).reshape(-1, outputChannels) / 32768.0
            outdata[:] = outdata * self.settings.serverOutputAudioGain  # GainはOutputのものをを流用
            # Monitorモードが有効の場合はサンプリングレートはmonitorデバイスが優先されているためリサンプリング不要
        except Exception as e:
            print("[Voice Changer][ServerDevice][audioMonitor_callback]  ex:", e)
            # import traceback
            # traceback.print_exc()

    def start(self):
        currentModelSamplingRate = -1
        while True:
            if self.settings.serverAudioStated == 0 or self.settings.serverInputDeviceId == -1:
                time.sleep(2)
            else:
                sd._terminate()
                sd._initialize()

                # Curret Device ID
                currentServerInputDeviceId = self.settings.serverInputDeviceId
                currentServerOutputDeviceId = self.settings.serverOutputDeviceId
                currentServerMonitorDeviceId = self.settings.serverMonitorDeviceId

                # Device 特定
                serverInputAudioDevice = self.getServerInputAudioDevice(self.settings.serverInputDeviceId)
                serverOutputAudioDevice = self.getServerOutputAudioDevice(self.settings.serverOutputDeviceId)
                serverMonitorAudioDevice = None
                if self.settings.serverMonitorDeviceId != -1:
                    serverMonitorAudioDevice = self.getServerOutputAudioDevice(self.settings.serverMonitorDeviceId)

                # Generate ExtraSetting
                inputExtraSetting = None
                outputExtraSetting = None
                if self.settings.exclusiveMode:
                    if "WASAPI" in serverInputAudioDevice.hostAPI:
                        inputExtraSetting = sd.WasapiSettings(exclusive=True)
                    if "WASAPI" in serverOutputAudioDevice.hostAPI:
                        outputExtraSetting = sd.WasapiSettings(exclusive=True)
                monitorExtraSetting = None
                if self.settings.exclusiveMode and serverMonitorAudioDevice is not None:
                    if "WASAPI" in serverMonitorAudioDevice.hostAPI:
                        monitorExtraSetting = sd.WasapiSettings(exclusive=True)

                print("Devices:")
                print("  [Input]:", serverInputAudioDevice, inputExtraSetting)
                print("  [Output]:", serverOutputAudioDevice, outputExtraSetting)
                print("  [Monitor]:", serverMonitorAudioDevice, monitorExtraSetting)

                # Deviceがなかったらいったんスリープ
                if serverInputAudioDevice is None or serverOutputAudioDevice is None:
                    print("serverInputAudioDevice or serverOutputAudioDevice is None")
                    time.sleep(2)
                    continue

                # サンプリングレート
                # 同一サンプリングレートに統一（変換時にサンプルが不足する場合があるため。パディング方法が明らかになれば、それぞれ設定できるかも）
                currentAudioSampleRate = self.settings.serverAudioSampleRate
                try:
                    currentModelSamplingRate = self.serverDeviceCallbacks.get_processing_sampling_rate()
                except Exception as e:
                    print("[Voice Changer] ex: get_processing_sampling_rate", e)
                    time.sleep(2)
                    continue

                self.settings.serverInputAudioSampleRate = currentAudioSampleRate
                self.settings.serverOutputAudioSampleRate = currentAudioSampleRate
                self.settings.serverMonitorAudioSampleRate = currentAudioSampleRate

                # Sample Rate Check
                inputAudioSampleRateAvailable = checkSamplingRate(self.settings.serverInputDeviceId, self.settings.serverInputAudioSampleRate, "input")
                outputAudioSampleRateAvailable = checkSamplingRate(self.settings.serverOutputDeviceId, self.settings.serverOutputAudioSampleRate, "output")
                monitorAudioSampleRateAvailable = checkSamplingRate(self.settings.serverMonitorDeviceId, self.settings.serverMonitorAudioSampleRate, "output") if serverMonitorAudioDevice else True

                print("Sample Rate:")
                print(f"  [Model]: {currentModelSamplingRate}")
                print(f"  [Input]: {self.settings.serverInputAudioSampleRate} -> {inputAudioSampleRateAvailable}")
                print(f"  [Output]: {self.settings.serverOutputAudioSampleRate} -> {outputAudioSampleRateAvailable}")
                if serverMonitorAudioDevice is not None:
                    print(f"  [Monitor]: {self.settings.serverMonitorAudioSampleRate} -> {monitorAudioSampleRateAvailable}")

                if inputAudioSampleRateAvailable and outputAudioSampleRateAvailable and monitorAudioSampleRateAvailable:
                    pass
                else:
                    print("Sample Rate is not supported by device:")
                    print("Checking Available Sample Rate:")
                    availableInputSampleRate = []
                    availableOutputSampleRate = []
                    availableMonitorSampleRate = []
                    for sr in SERVER_DEVICE_SAMPLE_RATES:
                        if checkSamplingRate(self.settings.serverInputDeviceId, sr, "input"):
                            availableInputSampleRate.append(sr)
                        if checkSamplingRate(self.settings.serverOutputDeviceId, sr, "output"):
                            availableOutputSampleRate.append(sr)
                        if serverMonitorAudioDevice is not None:
                            if checkSamplingRate(self.settings.serverMonitorDeviceId, sr, "output"):
                                availableMonitorSampleRate.append(sr)
                    print("Available Sample Rate:")
                    print(f"  [Input]: {availableInputSampleRate}")
                    print(f"  [Output]: {availableOutputSampleRate}")
                    if serverMonitorAudioDevice is not None:
                        print(f"  [Monitor]: {availableMonitorSampleRate}")

                    print("continue... ")
                    time.sleep(2)
                    continue

                self.serverDeviceCallbacks.setInputSamplingRate(self.settings.serverInputAudioSampleRate)
                self.serverDeviceCallbacks.setOutputSamplingRate(self.settings.serverOutputAudioSampleRate)

                # Blockサイズを計算
                currentInputChunkNum = self.settings.serverReadChunkSize
                # block_frame = currentInputChunkNum * 128
                block_frame = int(currentInputChunkNum * 128 * (self.settings.serverInputAudioSampleRate / 48000))

                sd.default.blocksize = block_frame

                # main loop
                try:
                    with sd.InputStream(
                        callback=self.audioInput_callback,
                        dtype="float32",
                        device=self.settings.serverInputDeviceId,
                        blocksize=block_frame,
                        samplerate=self.settings.serverInputAudioSampleRate,
                        channels=serverInputAudioDevice.maxInputChannels,
                        extra_settings=inputExtraSetting
                    ):
                        with sd.OutputStream(
                            callback=self.audioOutput_callback,
                            dtype="float32",
                            device=self.settings.serverOutputDeviceId,
                            blocksize=block_frame,
                            samplerate=self.settings.serverOutputAudioSampleRate,
                            channels=serverOutputAudioDevice.maxOutputChannels,
                            extra_settings=outputExtraSetting
                        ):
                            if self.settings.serverMonitorDeviceId != -1:
                                with sd.OutputStream(
                                    callback=self.audioMonitor_callback,
                                    dtype="float32",
                                    device=self.settings.serverMonitorDeviceId,
                                    blocksize=block_frame,
                                    samplerate=self.settings.serverMonitorAudioSampleRate,
                                    channels=serverMonitorAudioDevice.maxOutputChannels,
                                    extra_settings=monitorExtraSetting
                                ):
                                    while (
                                        self.settings.serverAudioStated == 1 and
                                        currentServerInputDeviceId == self.settings.serverInputDeviceId and
                                        currentServerOutputDeviceId == self.settings.serverOutputDeviceId and
                                        currentServerMonitorDeviceId == self.settings.serverMonitorDeviceId and
                                        currentModelSamplingRate == self.serverDeviceCallbacks.get_processing_sampling_rate() and
                                        currentInputChunkNum == self.settings.serverReadChunkSize and
                                        currentAudioSampleRate == self.settings.serverAudioSampleRate
                                    ):
                                        time.sleep(2)
                                        print(f"[Voice Changer] server audio performance {self.performance}")
                                        print(f"                status: started:{self.settings.serverAudioStated}, model_sr:{currentModelSamplingRate}, chunk:{currentInputChunkNum}")
                                        print(f"                input  : id:{self.settings.serverInputDeviceId}, sr:{self.settings.serverInputAudioSampleRate}, ch:{serverInputAudioDevice.maxInputChannels}")
                                        print(f"                output : id:{self.settings.serverOutputDeviceId}, sr:{self.settings.serverOutputAudioSampleRate}, ch:{serverOutputAudioDevice.maxOutputChannels}")
                                        print(f"                monitor: id:{self.settings.serverMonitorDeviceId}, sr:{self.settings.serverMonitorAudioSampleRate}, ch:{serverMonitorAudioDevice.maxOutputChannels}")
                            else:
                                while (
                                    self.settings.serverAudioStated == 1 and
                                    currentServerInputDeviceId == self.settings.serverInputDeviceId and
                                    currentServerOutputDeviceId == self.settings.serverOutputDeviceId and
                                    currentServerMonitorDeviceId == self.settings.serverMonitorDeviceId and
                                    currentModelSamplingRate == self.serverDeviceCallbacks.get_processing_sampling_rate() and
                                    currentInputChunkNum == self.settings.serverReadChunkSize and
                                    currentAudioSampleRate == self.settings.serverAudioSampleRate
                                ):
                                    time.sleep(2)
                                    print(f"[Voice Changer] server audio performance {self.performance}")
                                    print(f"                status: started:{self.settings.serverAudioStated}, model_sr:{currentModelSamplingRate}, chunk:{currentInputChunkNum}]")
                                    print(f"                input  : id:{self.settings.serverInputDeviceId}, sr:{self.settings.serverInputAudioSampleRate}, ch:{serverInputAudioDevice.maxInputChannels}")
                                    print(f"                output : id:{self.settings.serverOutputDeviceId}, sr:{self.settings.serverOutputAudioSampleRate}, ch:{serverOutputAudioDevice.maxOutputChannels}")
                except Exception as e:
                    print("[Voice Changer] processing, ex:", e)
                    time.sleep(2)

    def start2(self):
        # currentInputDeviceId = -1
        # currentOutputDeviceId = -1
        # currentInputChunkNum = -1
        currentModelSamplingRate = -1
        while True:
            if self.settings.serverAudioStated == 0 or self.settings.serverInputDeviceId == -1:
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
                        # dtype="int16",
                        # channels=[currentInputChannelNum, currentOutputChannelNum],
                    ):
                        pass
                    self.settings.serverInputAudioSampleRate = currentModelSamplingRate
                    self.serverDeviceCallbacks.setInputSamplingRate(currentModelSamplingRate)
                    self.serverDeviceCallbacks.setOutputSamplingRate(currentModelSamplingRate)
                    print(f"[Voice Changer] sample rate {self.settings.serverInputAudioSampleRate}")
                except Exception as e:
                    print("[Voice Changer] ex: fallback to device default samplerate", e)
                    print("[Voice Changer] device default samplerate", serverInputAudioDevice.default_samplerate)
                    self.settings.serverInputAudioSampleRate = round(serverInputAudioDevice.default_samplerate)
                    self.serverDeviceCallbacks.setInputSamplingRate(round(serverInputAudioDevice.default_samplerate))
                    self.serverDeviceCallbacks.setOutputSamplingRate(round(serverInputAudioDevice.default_samplerate))

                sd.default.samplerate = self.settings.serverInputAudioSampleRate
                sd.default.blocksize = block_frame
                # main loop
                try:
                    with sd.Stream(
                        callback=self.audio_callback,
                        # blocksize=block_frame,
                        # samplerate=vc.settings.serverInputAudioSampleRate,
                        dtype="float32",
                        # dtype="int16",
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
        try:
            audioinput, audiooutput = list_audio_device()
            self.serverAudioInputDevices = audioinput
            self.serverAudioOutputDevices = audiooutput
        except Exception as e:
            print(e)

        data["serverAudioInputDevices"] = self.serverAudioInputDevices
        data["serverAudioOutputDevices"] = self.serverAudioOutputDevices
        return data

    def update_settings(self, key: str, val: str | int | float):
        if key in EditableServerDeviceSettings["intData"]:
            setattr(self.settings, key, int(val))
        elif key in EditableServerDeviceSettings["floatData"]:
            setattr(self.settings, key, float(val))
        return self.get_info()
