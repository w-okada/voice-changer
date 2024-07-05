import warnings

import librosa
import numpy as np
from voice_changer.RVC.pitchExtractor import onnxcrepe


###############################################################################
# Constants
###############################################################################


# Minimum decibel level
MIN_DB = -100.

# Reference decibel level
REF_DB = 20.


###############################################################################
# A-weighted loudness
###############################################################################


def a_weighted(audio, sample_rate, hop_length=None, pad=True):
    """Retrieve the per-frame loudness"""

    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Convert to numpy
    audio = audio.squeeze(0)

    # Resample
    if sample_rate != onnxcrepe.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=onnxcrepe.SAMPLE_RATE)
        hop_length = int(hop_length * onnxcrepe.SAMPLE_RATE / sample_rate)

    # Cache weights
    if not hasattr(a_weighted, 'weights'):
        a_weighted.weights = perceptual_weights()

    # Take stft
    stft = librosa.stft(audio,
                        n_fft=onnxcrepe.WINDOW_SIZE,
                        hop_length=hop_length,
                        win_length=onnxcrepe.WINDOW_SIZE,
                        center=pad,
                        pad_mode='constant')

    # Compute magnitude on db scale
    db = librosa.amplitude_to_db(np.abs(stft))

    # Apply A-weighting
    weighted = db + a_weighted.weights

    # Threshold
    weighted[weighted < MIN_DB] = MIN_DB

    # Average over weighted frequencies
    return weighted.mean(axis=0, dtype=np.float32)[None]


def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(sr=onnxcrepe.SAMPLE_RATE,
                                          n_fft=onnxcrepe.WINDOW_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - REF_DB
