"""
Tests doing a spectrogram using an AudioSegment.
"""
import matplotlib
import platform
if platform.system() != "Windows":
    matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import read_from_file
import sys
import visualize

def test(seg):
    print("Doing a spectrogram...")
    print("  |-> Computing overlapping FFTs for first", visualize.VIS_MS, "ms...")
    hist_bins, times, amplitudes = seg[1:visualize.VIS_MS].spectrogram(window_length_s=0.03, overlap=1/4)
    amplitudes = 10 * np.log10(amplitudes + 1e-9)
    print("  |-> Plotting...")
    if os.environ.get('DISPLAY', False):
        plt.pcolormesh(times, hist_bins, amplitudes)
        plt.xlabel("Time in Seconds")
        plt.ylabel("Frequency in Hz")
        plt.show()

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
