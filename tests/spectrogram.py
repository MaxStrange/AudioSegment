"""
Tests doing a spectrogram using an AudioSegment.
"""
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
import read_from_file
import sys
import visualize

def test(seg):
    print("Doing a spectrogram...")
    print("  |-> Computing overlapping FFTs for first", visualize.VIS_MS, "ms...")
    hist_bins, times, amplitudes = seg[1:visualize.VIS_MS].spectrogram(window_length_s=0.03, overlap=0.5)
    hist_bins = hist_bins / 1000
    amplitudes = np.abs(amplitudes) / len(amplitudes)
    amplitudes = 10 * np.log10(amplitudes + 1e-9)
    print("  |-> Plotting...")
    x, y = np.mgrid[:len(times), :len(hist_bins)]
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, amplitudes)
    plt.show()

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
