"""
Tests creation of an FFT and plotting of it.
"""
import platform
import matplotlib
if platform.system() != "Windows":
    matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import os
import read_from_file
import sys

def visualize(spect, frequencies, title=""):
    '''Visualize the result of calling seg.filter_bank() for any number of filters'''
    i = 0
    for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
        plt.subplot(spect.shape[0], 1, index + 1)
        if i == 0:
            plt.title(title)
            i += 1
        plt.ylabel("Amp @ {0:.0f} Hz".format(freq))
        plt.plot(row)
    plt.show()

def test(seg):
    print("Applying filterbank...")
    spec, freqs = seg.filter_bank(nfilters=5)
    if os.environ.get('DISPLAY', False):
        print("  |-> Plotting...")
        visualize(spec, freqs)

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
