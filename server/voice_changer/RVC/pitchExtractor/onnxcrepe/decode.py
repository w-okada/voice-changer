import librosa
import numpy as np

from voice_changer.RVC.pitchExtractor import onnxcrepe

###############################################################################
# Probability sequence decoding methods
###############################################################################


def argmax(logits):
    """Sample observations by taking the argmax"""
    bins = logits.argmax(axis=1)

    # Convert to frequency in Hz
    return bins, onnxcrepe.convert.bins_to_frequency(bins)


def weighted_argmax(logits: np.ndarray):
    """Sample observations using weighted sum near the argmax"""
    # Find center of analysis window
    bins = logits.argmax(axis=1)

    return bins, _apply_weights(logits, bins)


def viterbi(logits):
    """Sample observations using viterbi decoding"""
    # Create viterbi transition matrix
    if not hasattr(viterbi, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        viterbi.transition = transition

    # Normalize logits (softmax)
    logits -= logits.max(axis=1)
    exp = np.exp(logits)
    probs = exp / np.sum(exp, axis=1)

    # Perform viterbi decoding
    bins = np.array([
        librosa.sequence.viterbi(sequence, viterbi.transition).astype(np.int64)
        for sequence in probs])

    # Convert to frequency in Hz
    return bins, onnxcrepe.convert.bins_to_frequency(bins)


def weighted_viterbi(logits):
    """Sample observations combining viterbi decoding and weighted argmax"""
    bins, _ = viterbi(logits)

    return bins, _apply_weights(logits, bins)


def _apply_weights(logits, bins):
    # Find bounds of analysis window
    start = np.maximum(0, bins - 4)
    end = np.minimum(logits.shape[1], bins + 5)

    # Mask out everything outside of window
    for batch in range(logits.shape[0]):
        for time in range(logits.shape[2]):
            logits[batch, :start[batch, time], time] = float('-inf')
            logits[batch, end[batch, time]:, time] = float('-inf')

    # Construct weights
    if not hasattr(_apply_weights, 'weights'):
        weights = onnxcrepe.convert.bins_to_cents(np.arange(360))
        _apply_weights.weights = weights[None, :, None]

    # Convert to probabilities (ReLU)
    probs = np.maximum(0, logits)

    # Apply weights
    cents = (_apply_weights.weights * probs).sum(axis=1) / probs.sum(axis=1)

    # Convert to frequency in Hz
    return onnxcrepe.convert.cents_to_frequency(cents)
