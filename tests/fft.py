"""
Tests creation of an FFT and plotting of it.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import common
import numpy as np
import os
import platform
import unittest

if os.environ.get('DISPLAY', False):
    import matplotlib
    if platform.system() != "Windows":
        matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt


class TestFFT(unittest.TestCase):
    """
    Tests for FFT functionality.
    """
    def test_pure_tone(self):
        # Generate a tone
        seconds = 1.0
        Fs = 16000.0
        Ftone = 400.0
        pure_seg = common.synthesize_pure_tone_segment(seconds, Fs, Ftone)

        # Do the FFT on the tone
        hist_bins, hist_vals = pure_seg.fft()
        hist_vals = np.abs(hist_vals) / len(hist_vals)

        # Check the peak is where you expect
        maxval = np.argmax(hist_vals)
        maxbin = hist_bins[maxval]
        self.assertAlmostEqual(maxbin, Ftone)

    @unittest.skipUnless(os.environ.get('DISPLAY', False), "Cannot do this test if no display.")
    def test_visualize(self):
        seg = audiosegment.from_file("furelise.wav")
        vis_ms = 3000
        hist_bins, hist_vals = seg[1:vis_ms].fft()
        hist_vals = np.abs(hist_vals) / len(hist_vals)

        # Now plot for human consumption
        plt.plot(hist_bins / 1000, hist_vals)
        plt.xlabel("kHz")
        plt.ylabel("dB")
        plt.show()


if __name__ == "__main__":
    unittest.main()
