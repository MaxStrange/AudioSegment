"""
Common functions for tests.
"""
import sys
import sys
sys.path.insert(0, '../')
import audiosegment
import math
import numpy as np
import os
import subprocess
import tempfile
import warnings

def is_playable(seg: audiosegment.AudioSegment) -> bool:
    """
    Attempts to play the given segment via Sox's play command and returns True
    if the process returns a zero return code, False otherwise.

    Only works if not running on Travis.
    """
    if os.environ.get("TRAVIS", False):
        # Can't use this method on Travis.
        return True

    with tempfile.NamedTemporaryFile('w+b', suffix='.wav', delete=True) as tmp:
        warnings.filterwarnings("ignore")
        seg.export(tmp.name, format="WAV")
        try:
            subprocess.run("play {}".format(tmp.name).split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1).check_returncode()
        except subprocess.TimeoutExpired:
            # This is normal for long files - we don't want to play the whole thing
            pass
        except Exception as e:
            print("Could not play the file. Error: {}".format(e))
            tmp.close()
            return False

    return True

def synthesize_pure_tone_array(duration_s: float, fs: float, ft: float, sample_width=2) -> np.ndarray:
    """
    Synthesize a pure tone of `ft` Hz, sampled at `fs` samples per second, of duration `duration_s` seconds.
    Return an array with the samples. Type of the array will be inferred by Numpy.
    """
    if fs < 0.5 * ft:
        raise ValueError("Need a sampling frequency of at least 2x the tone to prevent aliasing. Got fs {} and ft {}".format(fs, ft))

    nsteps = fs * duration_s
    times = np.arange(nsteps) / fs
    sinewave = np.sin(2.0 * np.pi * ft * times)

    return sinewave

def synthesize_pure_tone_segment(duration_s: float, fs: float, ft: float, dBFS=-6.0, sample_width=2) -> audiosegment.AudioSegment:
    """
    Synthesize a pure tone of `ft` Hz, sampled at `fs` samples per second, of duration `duration_s` seconds.
    Return an AudioSegment.
    """
    def dtype(arr):
        if sample_width == 1:
            return np.int8(arr)
        elif sample_width == 2:
            return np.int16(arr)
        elif sample_width == 4:
            return np.int32(arr)
        else:
            raise ValueError("Sample width of {} is not allowed.".format(sample_width))

    pure_tone = 100 * synthesize_pure_tone_array(duration_s, fs, ft)
    pure_seg = audiosegment.from_numpy_array(dtype(pure_tone), fs)
    curdb = pure_seg.dBFS
    pure_seg += (dBFS - curdb)
    return pure_seg


if __name__ == "__main__":
    seg = synthesize_pure_tone_segment(2, 16000, 400)
    print(seg.dBFS)
    seg = synthesize_pure_tone_segment(2, 32000, 8000, dBFS=-9)
    print(seg.dBFS)
