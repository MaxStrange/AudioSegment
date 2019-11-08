"""
Tests doing a spectrogram using an AudioSegment.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import platform
import numpy as np
import sys
import unittest

import os
if os.environ.get('DISPLAY', False):
    import matplotlib
    if platform.system() != "Windows":
        matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt


class TestSpectrogram(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DISPLAY', False), "Can't do this without a display.")
    def test_visualize(self):
        seg = audiosegment.from_file("furelise.wav")

        duration_s = 2.5
        hist_bins, times, amplitudes = seg.spectrogram(start_s=0, duration_s=duration_s, window_length_s=0.03, overlap=0.25)
        amplitudes = 10 * np.log10(amplitudes + 1e-9)

        plt.subplot(121)
        plt.pcolormesh(times, hist_bins, amplitudes)
        plt.xlabel("Time in Seconds")
        plt.ylabel("Frequency in Hz")

        hist_bins, times, amplitudes = seg.spectrogram(start_s=duration_s, duration_s=duration_s, window_length_s=0.03, overlap=0.25)
        times += duration_s
        amplitudes = 10 * np.log10(amplitudes + 1e-9)

        plt.subplot(122)
        plt.pcolormesh(times,hist_bins,amplitudes)
        plt.show()

    def test_spectrogram_functionality(self):
        """
        """
        seg = audiosegment.from_file("furelise.wav")

        duration_s = 2.5
        _hist_bins, _times, _amplitudes = seg.spectrogram(start_s=0, duration_s=duration_s, window_length_s=0.03, overlap=0.25)


if __name__ == "__main__":
    unittest.main()
