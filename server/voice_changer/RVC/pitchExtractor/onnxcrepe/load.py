import librosa
import numpy as np


def audio(filename):
    """Load audio from disk"""
    samples, sr = librosa.load(filename, sr=None)
    if len(samples.shape) > 1:
        # To mono
        samples = np.mean(samples, axis=1)

    return samples, sr
