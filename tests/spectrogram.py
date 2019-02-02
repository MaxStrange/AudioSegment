"""
Tests doing a spectrogram using an AudioSegment.
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
    print("Doing a spectrogram...")

    # Do the spectrogram
    duration_s = visualize.VIS_MS / 1000.0
    hist_bins, times, amplitudes = seg.spectrogram(start_s=0, duration_s=visualize.VIS_MS/1000.0, window_length_s=0.03, overlap=0.25)
    amplitudes = 10 * np.log10(amplitudes + 1e-9)

    print("Given a waveform of Fs {} and nsamples {}, and a spectrogram of window_length {} and overlap {}, we get {} hist bins and {} times.".format(
        seg.frame_rate, len(seg[1:visualize.VIS_MS]), 0.03, 1/4, hist_bins.shape, times.shape
    ))

    if os.environ.get('DISPLAY', False):
        plt.subplot(121)
        plt.pcolormesh(times, hist_bins, amplitudes)
        plt.xlabel("Time in Seconds")
        plt.ylabel("Frequency in Hz")

    hist_bins, times, amplitudes = seg.spectrogram(start_s=duration_s, duration_s=duration_s, window_length_s=0.03, overlap=0.25)
    times += duration_s
    amplitudes = 10 * np.log10(amplitudes + 1e-9)

    if os.environ.get('DISPLAY', False):
        plt.subplot(122)
        plt.pcolormesh(times,hist_bins,amplitudes)
        plt.show()

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
