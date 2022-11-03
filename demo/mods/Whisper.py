import whisper
import numpy as np
import torchaudio
from scipy.io.wavfile import write

_MODELS = {
    "tiny": "/whisper/tiny.pt",
    "base": "/whisper/base.pt",
    "small": "/whisper/small.pt",
    "medium": "/whisper/medium.pt",
}


class Whisper():
    def __init__(self):
        self.storedSizeFromTry = 0

    def loadModel(self, model):
        # self.model = whisper.load_model(_MODELS[model], device="cpu")
        self.model = whisper.load_model(_MODELS[model])
        self.data = np.zeros(1).astype(np.float)
    
    def addData(self, unpackedData):
        self.data = np.concatenate([self.data, unpackedData], 0)

    def transcribe(self, audio):
        received_data_file = "received_data.wav"
        write(received_data_file, 24000, self.data.astype(np.int16))
        source, sr = torchaudio.load(received_data_file) 
        target = torchaudio.functional.resample(source, 24000, 16000)
        result = self.model.transcribe(received_data_file)
        print("WHISPER1:::", result["text"])
        print("WHISPER2:::", result["segments"])
        self.data = np.zeros(1).astype(np.float)
        return result["text"]

