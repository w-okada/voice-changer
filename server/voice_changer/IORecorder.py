import wave
import os
from mods.log_control import VoiceChangaerLogger

logger = VoiceChangaerLogger.get_instance().getLogger()


class IORecorder:
    def __init__(self, inputFilename: str, outputFilename: str, inputSamplingRate: int, outputSamplingRate: int):
        self._clearFile(inputFilename)
        self._clearFile(outputFilename)

        self.fi = wave.open(inputFilename, "wb")
        self.fi.setnchannels(1)
        self.fi.setsampwidth(2)
        self.fi.setframerate(inputSamplingRate)

        self.fo = wave.open(outputFilename, "wb")
        self.fo.setnchannels(1)
        self.fo.setsampwidth(2)
        self.fo.setframerate(outputSamplingRate)

    def _clearFile(self, filename: str):
        if os.path.exists(filename):
            logger.info(f"[IORecorder] delete old analyze file. {filename}")
            os.remove(filename)
        else:
            logger.info(f"[IORecorder] old analyze file not exist. {filename}")

    def writeInput(self, wav):
        self.fi.writeframes(wav)

    def writeOutput(self, wav):
        self.fo.writeframes(wav)

    def close(self):
        self.fi.close()
        self.fo.close()
