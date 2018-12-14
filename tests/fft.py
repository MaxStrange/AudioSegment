"""
Tests creation of an FFT and plotting of it.
"""
import platform
import os
if os.environ.get('DISPLAY', False):
    import matplotlib
    if platform.system() != "Windows":
        matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
import numpy as np
import read_from_file
import sys
import visualize

def test(seg):
    print("Doing FFT and plotting the histogram...")
    print("  |-> Computing the FFT on the first", visualize.VIS_MS, "ms...")
    hist_bins, hist_vals = seg[1:visualize.VIS_MS].fft()
    hist_vals = np.abs(hist_vals) / len(hist_vals)
    print("  |-> Plotting...")
    if os.environ.get('DISPLAY', False):
        plt.plot(hist_bins / 1000, hist_vals)
        plt.xlabel("kHz")
        plt.ylabel("dB")
        plt.show()

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
