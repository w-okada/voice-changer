from dataclasses import dataclass, asdict

import numpy as np
from const import SERVER_DEVICE_SAMPLE_RATES

from queue import Queue
from mods.log_control import VoiceChangaerLogger

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

logger = VoiceChangaerLogger.get_instance().getLogger()

# See https://github.com/w-okada/voice-changer/issues/620
LocalServerDeviceMode: TypeAlias = Literal[
    "NoMonitorSeparate",
    "WithMonitorStandard",
    "WithMonitorAllSeparate",
]


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
    serverMonitorAudioGain: float = 1.0

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
        "serverMonitorAudioGain",
    ],
    "boolData": ["exclusiveMode"],
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
        self.performance = []

        # setting change確認用
        self.currentServerInputDeviceId = -1
        self.currentServerOutputDeviceId = -1
        self.currentServerMonitorDeviceId = -1
        self.currentModelSamplingRate = -1
        self.currentInputChunkNum = -1
        self.currentAudioSampleRate = -1

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

    ###########################################
    # Callback Section
    ###########################################

    def _processData(self, indata: np.ndarray):
        indata = indata * self.settings.serverInputAudioGain
        unpackedData = librosa.to_mono(indata.T) * 32768.0
        unpackedData = unpackedData.astype(np.int16)
        out_wav, times = self.serverDeviceCallbacks.on_request(unpackedData)
        return out_wav, times

    def _processDataWithTime(self, indata: np.ndarray):
        with Timer("all_inference_time") as t:
            out_wav, times = self._processData(indata)
        all_inference_time = t.secs
        self.performance = [all_inference_time] + times
        self.serverDeviceCallbacks.emitTo(self.performance)
        self.performance = [round(x * 1000) for x in self.performance]
        return out_wav

    def audio_callback_outQueue(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        try:
            out_wav = self._processDataWithTime(indata)

            self.outQueue.put(out_wav)
            outputChannels = outdata.shape[1]  # Monitorへのアウトプット
            outdata[:] = np.repeat(out_wav, outputChannels).reshape(-1, outputChannels) / 32768.0
            outdata[:] = outdata * self.settings.serverMonitorAudioGain
        except Exception as e:
            print("[Voice Changer] ex:", e)

    def audioInput_callback_outQueue(self, indata: np.ndarray, frames, times, status):
        try:
            out_wav = self._processDataWithTime(indata)
            self.outQueue.put(out_wav)
        except Exception as e:
            print("[Voice Changer][ServerDevice][audioInput_callback] ex:", e)
            # import traceback
            # traceback.print_exc()

    def audioInput_callback_outQueue_monQueue(self, indata: np.ndarray, frames, times, status):
        try:
            out_wav = self._processDataWithTime(indata)
            self.outQueue.put(out_wav)
            self.monQueue.put(out_wav)
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
            outdata[:] = outdata * self.settings.serverMonitorAudioGain
        except Exception as e:
            print("[Voice Changer][ServerDevice][audioMonitor_callback]  ex:", e)
            # import traceback
            # traceback.print_exc()

    ###########################################
    # Main Loop Section
    ###########################################
    def checkSettingChanged(self):
        if self.settings.serverAudioStated != 1:
            print(f"serverAudioStarted Changed: {self.settings.serverAudioStated}")
            return True
        elif self.currentServerInputDeviceId != self.settings.serverInputDeviceId:
            print(f"serverInputDeviceId Changed: {self.currentServerInputDeviceId} -> {self.settings.serverInputDeviceId}")
            return True
        elif self.currentServerOutputDeviceId != self.settings.serverOutputDeviceId:
            print(f"serverOutputDeviceId Changed: {self.currentServerOutputDeviceId} -> {self.settings.serverOutputDeviceId}")
            return True
        elif self.currentServerMonitorDeviceId != self.settings.serverMonitorDeviceId:
            print(f"serverMonitorDeviceId Changed: {self.currentServerMonitorDeviceId} -> {self.settings.serverMonitorDeviceId}")
            return True
        elif self.currentModelSamplingRate != self.serverDeviceCallbacks.get_processing_sampling_rate():
            print(f"currentModelSamplingRate Changed: {self.currentModelSamplingRate} -> {self.serverDeviceCallbacks.get_processing_sampling_rate()}")
            return True
        elif self.currentInputChunkNum != self.settings.serverReadChunkSize:
            print(f"currentInputChunkNum Changed: {self.currentInputChunkNum} -> {self.settings.serverReadChunkSize}")
            return True
        elif self.currentAudioSampleRate != self.settings.serverAudioSampleRate:
            print(f"currentAudioSampleRate Changed: {self.currentAudioSampleRate} -> {self.settings.serverAudioSampleRate}")
            return True
        else:
            return False

    def runNoMonitorSeparate(self, block_frame: int, inputMaxChannel: int, outputMaxChannel: int, inputExtraSetting, outputExtraSetting):
        with sd.InputStream(callback=self.audioInput_callback_outQueue, dtype="float32", device=self.settings.serverInputDeviceId, blocksize=block_frame, samplerate=self.settings.serverInputAudioSampleRate, channels=inputMaxChannel, extra_settings=inputExtraSetting):
            with sd.OutputStream(callback=self.audioOutput_callback, dtype="float32", device=self.settings.serverOutputDeviceId, blocksize=block_frame, samplerate=self.settings.serverOutputAudioSampleRate, channels=outputMaxChannel, extra_settings=outputExtraSetting):
                while True:
                    changed = self.checkSettingChanged()
                    if changed:
                        break
                    time.sleep(2)
                    print(f"[Voice Changer] server audio performance {self.performance}")
                    print(f"                status: started:{self.settings.serverAudioStated}, model_sr:{self.currentModelSamplingRate}, chunk:{self.currentInputChunkNum}")
                    print(f"                input  : id:{self.settings.serverInputDeviceId}, sr:{self.settings.serverInputAudioSampleRate}, ch:{inputMaxChannel}")
                    print(f"                output : id:{self.settings.serverOutputDeviceId}, sr:{self.settings.serverOutputAudioSampleRate}, ch:{outputMaxChannel}")
                    # print(f"                monitor: id:{self.settings.serverMonitorDeviceId}, sr:{self.settings.serverMonitorAudioSampleRate}, ch:{self.serverMonitorAudioDevice.maxOutputChannels}")

    def runWithMonitorStandard(self, block_frame: int, inputMaxChannel: int, outputMaxChannel: int, monitorMaxChannel: int, inputExtraSetting, outputExtraSetting, monitorExtraSetting):
        with sd.Stream(callback=self.audio_callback_outQueue, dtype="float32", device=(self.settings.serverInputDeviceId, self.settings.serverMonitorDeviceId), blocksize=block_frame, samplerate=self.settings.serverInputAudioSampleRate, channels=(inputMaxChannel, monitorMaxChannel), extra_settings=[inputExtraSetting, monitorExtraSetting]):
            with sd.OutputStream(callback=self.audioOutput_callback, dtype="float32", device=self.settings.serverOutputDeviceId, blocksize=block_frame, samplerate=self.settings.serverOutputAudioSampleRate, channels=outputMaxChannel, extra_settings=outputExtraSetting):
                while True:
                    changed = self.checkSettingChanged()
                    if changed:
                        break
                    time.sleep(2)
                    print(f"[Voice Changer] server audio performance {self.performance}")
                    print(f"                status: started:{self.settings.serverAudioStated}, model_sr:{self.currentModelSamplingRate}, chunk:{self.currentInputChunkNum}")
                    print(f"                input  : id:{self.settings.serverInputDeviceId}, sr:{self.settings.serverInputAudioSampleRate}, ch:{inputMaxChannel}")
                    print(f"                output : id:{self.settings.serverOutputDeviceId}, sr:{self.settings.serverOutputAudioSampleRate}, ch:{outputMaxChannel}")
                    print(f"                monitor: id:{self.settings.serverMonitorDeviceId}, sr:{self.settings.serverMonitorAudioSampleRate}, ch:{monitorMaxChannel}")

    def runWithMonitorAllSeparate(self, block_frame: int, inputMaxChannel: int, outputMaxChannel: int, monitorMaxChannel: int, inputExtraSetting, outputExtraSetting, monitorExtraSetting):
        with sd.InputStream(callback=self.audioInput_callback_outQueue_monQueue, dtype="float32", device=self.settings.serverInputDeviceId, blocksize=block_frame, samplerate=self.settings.serverInputAudioSampleRate, channels=inputMaxChannel, extra_settings=inputExtraSetting):
            with sd.OutputStream(callback=self.audioOutput_callback, dtype="float32", device=self.settings.serverOutputDeviceId, blocksize=block_frame, samplerate=self.settings.serverOutputAudioSampleRate, channels=outputMaxChannel, extra_settings=outputExtraSetting):
                with sd.OutputStream(callback=self.audioMonitor_callback, dtype="float32", device=self.settings.serverMonitorDeviceId, blocksize=block_frame, samplerate=self.settings.serverMonitorAudioSampleRate, channels=monitorMaxChannel, extra_settings=monitorExtraSetting):
                    while True:
                        changed = self.checkSettingChanged()
                        if changed:
                            break
                        time.sleep(2)
                        print(f"[Voice Changer] server audio performance {self.performance}")
                        print(f"                status: started:{self.settings.serverAudioStated}, model_sr:{self.currentModelSamplingRate}, chunk:{self.currentInputChunkNum}")
                        print(f"                input  : id:{self.settings.serverInputDeviceId}, sr:{self.settings.serverInputAudioSampleRate}, ch:{inputMaxChannel}")
                        print(f"                output : id:{self.settings.serverOutputDeviceId}, sr:{self.settings.serverOutputAudioSampleRate}, ch:{outputMaxChannel}")
                        print(f"                monitor: id:{self.settings.serverMonitorDeviceId}, sr:{self.settings.serverMonitorAudioSampleRate}, ch:{monitorMaxChannel}")

    ###########################################
    # Start Section
    ###########################################
    def start(self):
        self.currentModelSamplingRate = -1
        while True:
            if self.settings.serverAudioStated == 0 or self.settings.serverInputDeviceId == -1:
                sd._terminate()
                sd._initialize()
                time.sleep(2)
            else:
                sd._terminate()
                sd._initialize()

                # Curret Device ID
                self.currentServerInputDeviceId = self.settings.serverInputDeviceId
                self.currentServerOutputDeviceId = self.settings.serverOutputDeviceId
                self.currentServerMonitorDeviceId = self.settings.serverMonitorDeviceId

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
                self.currentAudioSampleRate = self.settings.serverAudioSampleRate
                try:
                    self.currentModelSamplingRate = self.serverDeviceCallbacks.get_processing_sampling_rate()
                except Exception as e:
                    print("[Voice Changer] ex: get_processing_sampling_rate", e)
                    time.sleep(2)
                    continue

                self.settings.serverInputAudioSampleRate = self.currentAudioSampleRate
                self.settings.serverOutputAudioSampleRate = self.currentAudioSampleRate
                self.settings.serverMonitorAudioSampleRate = self.currentAudioSampleRate

                # Sample Rate Check
                inputAudioSampleRateAvailable = checkSamplingRate(self.settings.serverInputDeviceId, self.settings.serverInputAudioSampleRate, "input")
                outputAudioSampleRateAvailable = checkSamplingRate(self.settings.serverOutputDeviceId, self.settings.serverOutputAudioSampleRate, "output")
                monitorAudioSampleRateAvailable = checkSamplingRate(self.settings.serverMonitorDeviceId, self.settings.serverMonitorAudioSampleRate, "output") if serverMonitorAudioDevice else True

                print("Sample Rate:")
                print(f"  [Model]: {self.currentModelSamplingRate}")
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
                self.currentInputChunkNum = self.settings.serverReadChunkSize
                # block_frame = currentInputChunkNum * 128
                block_frame = int(self.currentInputChunkNum * 128 * (self.settings.serverInputAudioSampleRate / 48000))

                sd.default.blocksize = block_frame

                # main loop
                try:
                    # See https://github.com/w-okada/voice-changer/issues/620
                    def judgeServerDeviceMode() -> LocalServerDeviceMode:
                        if self.settings.serverMonitorDeviceId == -1:
                            return "NoMonitorSeparate"
                        else:
                            if serverInputAudioDevice.hostAPI == serverOutputAudioDevice.hostAPI and serverInputAudioDevice.hostAPI == serverMonitorAudioDevice.hostAPI:  # すべて同じ
                                return "WithMonitorStandard"
                            elif serverInputAudioDevice.hostAPI != serverOutputAudioDevice.hostAPI and serverInputAudioDevice.hostAPI != serverMonitorAudioDevice.hostAPI and serverOutputAudioDevice.hostAPI != serverMonitorAudioDevice.hostAPI:  # すべて違う
                                return "WithMonitorAllSeparate"
                            elif serverInputAudioDevice.hostAPI == serverOutputAudioDevice.hostAPI:  # in/outだけが同じ
                                return "WithMonitorAllSeparate"
                            elif serverInputAudioDevice.hostAPI == serverMonitorAudioDevice.hostAPI:  # in/monだけが同じ
                                return "WithMonitorStandard"
                            elif serverOutputAudioDevice.hostAPI == serverMonitorAudioDevice.hostAPI:  # out/monだけが同じ
                                return "WithMonitorAllSeparate"
                            else:
                                raise RuntimeError(f"Cannot JudgeServerMode, in:{serverInputAudioDevice.hostAPI}, mon:{serverMonitorAudioDevice.hostAPI}, out:{serverOutputAudioDevice.hostAPI}")

                    serverDeviceMode = judgeServerDeviceMode()
                    if serverDeviceMode == "NoMonitorSeparate":
                        self.runNoMonitorSeparate(block_frame, serverInputAudioDevice.maxInputChannels, serverOutputAudioDevice.maxOutputChannels, inputExtraSetting, outputExtraSetting)
                    elif serverDeviceMode == "WithMonitorStandard":
                        self.runWithMonitorStandard(block_frame, serverInputAudioDevice.maxInputChannels, serverOutputAudioDevice.maxOutputChannels, serverMonitorAudioDevice.maxOutputChannels, inputExtraSetting, outputExtraSetting, monitorExtraSetting)
                    elif serverDeviceMode == "WithMonitorAllSeparate":
                        self.runWithMonitorAllSeparate(block_frame, serverInputAudioDevice.maxInputChannels, serverOutputAudioDevice.maxOutputChannels, serverMonitorAudioDevice.maxOutputChannels, inputExtraSetting, outputExtraSetting, monitorExtraSetting)
                    else:
                        raise RuntimeError(f"Unknown ServerDeviceMode: {serverDeviceMode}")

                except Exception as e:
                    print("[Voice Changer] processing, ex:", e)
                    import traceback

                    traceback.print_exc()
                    time.sleep(2)

    ###########################################
    # Info Section
    ###########################################
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
