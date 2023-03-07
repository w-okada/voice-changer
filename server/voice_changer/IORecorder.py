import wave
import os


class IORecorder:

    def __init__(self, inputFilename: str, outputFilename: str, samplingRate: int):

        self._clearFile(inputFilename)
        self._clearFile(outputFilename)

        self.fi = wave.open(inputFilename, 'wb')
        self.fi.setnchannels(1)
        self.fi.setsampwidth(2)
        self.fi.setframerate(samplingRate)

        self.fo = wave.open(outputFilename, 'wb')
        self.fo.setnchannels(1)
        self.fo.setsampwidth(2)
        self.fo.setframerate(samplingRate)

    def _clearFile(self, filename: str):
        if os.path.exists(filename):
            print("[IORecorder] delete old analyze file.", filename)
            os.remove(filename)
        else:
            print("[IORecorder] old analyze file not exist.", filename)

    def writeInput(self, wav):
        self.fi.writeframes(wav)

    def writeOutput(self, wav):
        self.fo.writeframes(wav)

    def close(self):
        self.fi.close()
        self.fo.close()
