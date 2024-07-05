import librosa
import numpy as np

from voice_changer.RVC.pitchExtractor import onnxcrepe

__all__ = ['CENTS_PER_BIN',
           'MAX_FMAX',
           'PITCH_BINS',
           'SAMPLE_RATE',
           'WINDOW_SIZE',
           'UNVOICED',
           'predict',
           'preprocess',
           'infer',
           'postprocess',
           'resample']

###############################################################################
# Constants
###############################################################################


CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples
UNVOICED = np.nan


###############################################################################
# Crepe pitch prediction
###############################################################################


def predict(session,
            audio: np.ndarray,
            sample_rate: int,
            precision=None,
            fmin=50.,
            fmax=MAX_FMAX,
            decoder=onnxcrepe.decode.weighted_viterbi,
            return_periodicity=False,
            batch_size=None,
            pad=True) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Performs pitch estimation

    Arguments
        session (onnxcrepe.CrepeInferenceSession)
            An onnxruntime.InferenceSession holding the CREPE model
        audio (numpy.ndarray [shape=(n_samples,)])
            The audio signal
        sample_rate (int)
            The sampling rate in Hz
        precision (float)
            The precision in milliseconds, i.e. the length of each frame
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_periodicity (bool)
            Whether to also return the network confidence
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        pitch (numpy.ndarray [shape=(1, 1 + int(time // precision))])
        (Optional) periodicity (numpy.ndarray
                                [shape=(1, 1 + int(time // precision))])
    """

    results = []

    # Preprocess audio
    generator = preprocess(audio,
                           sample_rate,
                           precision,
                           batch_size,
                           pad)
    for frames in generator:

        # Infer independent probabilities for each pitch bin
        probabilities = infer(session, frames)  # shape=(batch, 360)

        probabilities = probabilities.transpose(1, 0)[None]  # shape=(1, 360, batch)

        # Convert probabilities to F0 and periodicity
        result = postprocess(probabilities,
                             fmin,
                             fmax,
                             decoder,
                             return_periodicity)

        # Place on same device as audio to allow very long inputs
        if isinstance(result, tuple):
            result = (result[0], result[1])

        results.append(result)

    # Split pitch and periodicity
    if return_periodicity:
        pitch, periodicity = zip(*results)
        return np.concatenate(pitch, axis=1), np.concatenate(periodicity, axis=1)

    # Concatenate
    return np.concatenate(results, axis=1)


def preprocess(audio: np.ndarray,
               sample_rate: int,
               precision=None,
               batch_size=None,
               pad=True):
    """Convert audio to model input

    Arguments
        audio (numpy.ndarray [shape=(time,)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        precision (float)
            The precision in milliseconds, i.e. the length of each frame
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        frames (numpy.ndarray [shape=(1 + int(time // precision), 1024)])
    """
    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    # Default hop length of 10 ms
    hop_length = SAMPLE_RATE / 100 if precision is None else SAMPLE_RATE * precision / 1000

    # Get total number of frames

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.shape[0] / hop_length)
        audio = np.pad(
            audio,
            (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else:
        total_frames = 1 + int((audio.shape[0] - WINDOW_SIZE) / hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):
        # Batch indices
        start = max(0, int(i * hop_length))
        end = min(audio.shape[0],
                  int((i + batch_size - 1) * hop_length) + WINDOW_SIZE)

        # Chunk
        n_bytes = audio.strides[-1]
        frames = np.lib.stride_tricks.as_strided(
            audio[start:end],
            shape=((end - start - WINDOW_SIZE) // int(hop_length) + 1, WINDOW_SIZE),
            strides=(int(hop_length) * n_bytes, n_bytes))  # shape=(batch, 1024)

        # Note:
        # Z-score standardization operations originally located here
        # (https://github.com/maxrmorrison/torchcrepe/blob/master/torchcrepe/core.py#L692)
        # are wrapped into the ONNX models for hardware acceleration.

        yield frames


def infer(session, frames) -> np.ndarray:
    """Forward pass through the model

    Arguments
        session (onnxcrepe.CrepeInferenceSession)
            An onnxruntime.InferenceSession holding the CREPE model
        frames (numpy.ndarray [shape=(time / precision, 1024)])
            The network input

    Returns
        logits (numpy.ndarray [shape=(1 + int(time // precision), 360)])
    """
    # Apply model
    return session.run(None, {'frames': frames})[0]


def postprocess(probabilities,
                fmin=0.,
                fmax=MAX_FMAX,
                decoder=onnxcrepe.decode.weighted_viterbi,
                return_periodicity=False):
    """Convert model output to F0 and periodicity

    Arguments
        probabilities (numpy.ndarray [shape=(1, 360, time / precision)])
            The probabilities for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_periodicity (bool)
            Whether to also return the network confidence

    Returns
        pitch (numpy.ndarray [shape=(1, 1 + int(time // precision))])
        periodicity (numpy.ndarray [shape=(1, 1 + int(time // precision))])
    """
    # Convert frequency range to pitch bin range
    minidx = onnxcrepe.convert.frequency_to_bins(fmin)
    maxidx = onnxcrepe.convert.frequency_to_bins(fmax, np.ceil)

    # Remove frequencies outside allowable range
    probabilities[:, :minidx] = float('-inf')
    probabilities[:, maxidx:] = float('-inf')

    # Perform argmax or viterbi sampling
    bins, pitch = decoder(probabilities)

    if not return_periodicity:
        return pitch

    # Compute periodicity from probabilities and decoded pitch bins
    return pitch, periodicity(probabilities, bins)


###############################################################################
# Utilities
###############################################################################


def periodicity(probabilities: np.ndarray, bins: np.ndarray):
    """Computes the periodicity from the network output and pitch bins"""
    # shape=(time / precision, 360)
    probs_stacked = probabilities.transpose(0, 2, 1).reshape(-1, PITCH_BINS)
    # shape=(time / precision, 1)
    bins_stacked = bins.reshape(-1, 1).astype(np.int64, copy=False)

    # Use maximum logit over pitch bins as periodicity
    periodicity = np.take_along_axis(probs_stacked, bins_stacked, axis=1)

    # shape=(batch, time / precision)
    return periodicity.reshape(probabilities.shape[0], probabilities.shape[2])


def resample(audio: np.ndarray, sample_rate: int):
    """Resample audio"""
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=onnxcrepe.SAMPLE_RATE)
