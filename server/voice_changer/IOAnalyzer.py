import os
import numpy as np
import pylab
import librosa
import librosa.display
import pyworld as pw


class IOAnalyzer:

    def _get_f0_dio(self, y, sr):
        _f0, time = pw.dio(y, sr, frame_period=5)
        f0 = pw.stonemask(y, _f0, time, sr)
        time = np.linspace(0, y.shape[0] / sr, len(time))
        return f0, time

    def _get_f0_harvest(self, y, sr):
        _f0, time = pw.harvest(y, sr, frame_period=5)
        f0 = pw.stonemask(y, _f0, time, sr)
        time = np.linspace(0, y.shape[0] / sr, len(time))
        return f0, time

    def analyze(self, inputDataFile: str, dioImageFile: str, harvestImageFile: str, samplingRate: int):
        y, sr = librosa.load(inputDataFile, samplingRate)
        y = y.astype(np.float64)
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, win_length=2048, hop_length=128)), ref=np.max)
        f0_dio, times = self._get_f0_dio(y, sr=samplingRate)
        f0_harvest, times = self._get_f0_harvest(y, sr=samplingRate)

        pylab.close()
        HOP_LENGTH = 128
        img = librosa.display.specshow(spec, sr=samplingRate, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', )
        pylab.plot(times, f0_dio, label='f0', color=(0, 1, 1, 0.6), linewidth=3)
        pylab.savefig(dioImageFile)

        pylab.close()
        HOP_LENGTH = 128
        img = librosa.display.specshow(spec, sr=samplingRate, hop_length=HOP_LENGTH, x_axis='time', y_axis='log', )
        pylab.plot(times, f0_harvest, label='f0', color=(0, 1, 1, 0.6), linewidth=3)
        pylab.savefig(harvestImageFile)
