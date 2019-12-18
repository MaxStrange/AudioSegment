"""
Convenience functions for using Numpy/Scipy filters in the audio domain.
"""
import ctypes
import numpy as np
import os

_mydir = os.path.dirname(__file__)
_so_path = os.path.join(_mydir, "signal.so")

# Try to import Scipy. If not present, we can't use these functions.
try:
    import scipy.signal as signal
    scipy_imported = True
except ImportError:
    scipy_imported = False


def bandpass_filter(data, low, high, fs, order=5):
    """
    Does a bandpass filter over the given data.

    :param data: The data (numpy array) to be filtered.
    :param low: The low cutoff in Hz.
    :param high: The high cutoff in Hz.
    :param fs: The sample rate (in Hz) of the data.
    :param order: The order of the filter. The higher the order, the tighter the roll-off.
    :returns: Filtered data (numpy array).
    """
    if not scipy_imported:
        raise NotImplementedError("This function is unusable without Scipy")

    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y

def lowpass_filter(data, cutoff, fs, order=5):
    """
    Does a lowpass filter over the given data.

    :param data: The data (numpy array) to be filtered.
    :param cutoff: The high cutoff in Hz.
    :param fs: The sample rate in Hz of the data.
    :param order: The order of the filter. The higher the order, the tighter the roll-off.
    :returns: Filtered data (numpy array).
    """
    if not scipy_imported:
        raise NotImplementedError("This function is unusable without Scipy")

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.lfilter(b, a, data)
    return y
