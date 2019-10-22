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
import read_from_file
import sys

def visualize(spect, frequencies, title=""):
    """Visualize the result of calling seg.filter_bank() for any number of filters"""
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
    # If we are running in Travis and are Python 3.4, we can't use this function,
    # so let's skip the test
    try:
        import librosa
        cannot_import_librosa = False
    except ImportError:
        cannot_import_librosa = True
    if cannot_import_librosa and os.environ.get("TRAVIS", False) and sys.version_info.minor < 5:
        print("Skipping filterbank test for this version of python, as we cannot import librosa.")
        return

    seg = seg[:25000]
    print("Applying filterbank...")
    spec, freqs = seg.filter_bank(nfilters=5, mode='log')
    if os.environ.get('DISPLAY', False):
        print("  |-> Plotting...")
        visualize(spec, freqs)
    print("Now trying linear")
    spec, freqs = seg.filter_bank(nfilters=5, mode='mel')
    print("Now trying log")
    spec, freqs = seg.filter_bank(nfilters=5, mode='log')

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
