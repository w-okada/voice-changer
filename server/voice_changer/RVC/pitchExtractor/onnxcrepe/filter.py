import numpy as np


###############################################################################
# Sequence filters
###############################################################################


def mean(signals, win_length=9):
    """Averave filtering for signals containing nan values

    Arguments
        signals (numpy.ndarray (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (numpy.ndarray (shape=(batch, time)))
    """
    return nanfilter(signals, win_length, nanmean)


def median(signals, win_length):
    """Median filtering for signals containing nan values

    Arguments
        signals (numpy.ndarray (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window

    Returns
        filtered (numpy.ndarray (shape=(batch, time)))
    """
    return nanfilter(signals, win_length, nanmedian)


###############################################################################
# Utilities
###############################################################################


def nanfilter(signals, win_length, filter_fn):
    """Filters a sequence, ignoring nan values

    Arguments
        signals (numpy.ndarray (shape=(batch, time)))
            The signals to filter
        win_length
            The size of the analysis window
        filter_fn (function)
            The function to use for filtering

    Returns
        filtered (numpy.ndarray (shape=(batch, time)))
    """
    # Output buffer
    filtered = np.empty_like(signals)

    # Loop over frames
    for i in range(signals.shape[1]):

        # Get analysis window bounds
        start = max(0, i - win_length // 2)
        end = min(signals.shape[1], i + win_length // 2 + 1)

        # Apply filter to window
        filtered[:, i] = filter_fn(signals[:, start:end])

    return filtered


def nanmean(signals):
    """Computes the mean, ignoring nans

    Arguments
        signals (numpy.ndarray [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (numpy.ndarray [shape=(batch, time)])
    """
    signals = signals.clone()

    # Find nans
    nans = np.isnan(signals)

    # Set nans to 0.
    signals[nans] = 0.

    # Compute average
    return signals.sum(axis=1) / (~nans).astype(np.float32).sum(axis=1)


def nanmedian(signals):
    """Computes the median, ignoring nans

    Arguments
        signals (numpy.ndarray [shape=(batch, time)])
            The signals to filter

    Returns
        filtered (numpy.ndarray [shape=(batch, time)])
    """
    # Find nans
    nans = np.isnan(signals)

    # Compute median for each slice
    medians = [nanmedian1d(signal[~nan]) for signal, nan in zip(signals, nans)]

    # Stack results
    return np.array(medians, dtype=signals.dtype)


def nanmedian1d(signal):
    """Computes the median. If signal is empty, returns torch.nan

    Arguments
        signal (numpy.ndarray [shape=(time,)])

    Returns
        median (numpy.ndarray [shape=(1,)])
    """
    return np.median(signal) if signal.size else np.nan
