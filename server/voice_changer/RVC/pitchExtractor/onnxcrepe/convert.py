import numpy as np
import scipy

from voice_changer.RVC.pitchExtractor import onnxcrepe


###############################################################################
# Pitch unit conversions
###############################################################################


def bins_to_cents(bins, apply_dither=False):
    """Converts pitch bins to cents"""
    cents = onnxcrepe.CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise (disabled by default)
    return dither(cents) if apply_dither else cents


def bins_to_frequency(bins, apply_dither=False):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins, apply_dither=apply_dither))


def cents_to_bins(cents, quantize_fn=np.floor):
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / onnxcrepe.CENTS_PER_BIN
    return quantize_fn(bins).astype(np.int64, copy=False)


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=np.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * np.log2(frequency / 10.)


###############################################################################
# Utilities
###############################################################################


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    noise = scipy.stats.triang.rvs(c=0.5,
                                   loc=-onnxcrepe.CENTS_PER_BIN,
                                   scale=2 * onnxcrepe.CENTS_PER_BIN,
                                   size=cents.shape)
    return cents + noise.astype(cents.dtype)
