"""
Tests creation of an FFT and plotting of it.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import platform
import os
import unittest

if os.environ.get('DISPLAY', False):
    import matplotlib
    if platform.system() != "Windows":
        matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt

# If we are running in Travis and are Python 3.4, we can't use this function,
# so let's skip the test
skip = False
try:
    import librosa
    cannot_import_librosa = False
except ImportError:
    cannot_import_librosa = True
if cannot_import_librosa and os.environ.get("TRAVIS", False) and sys.version_info.minor < 5:
    print("Skipping filterbank test for this version of python, as we cannot import librosa.")
    skip = True

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

@unittest.skipIf(skip, "Can't run this test for this version of Python.")
class TestFilterBank(unittest.TestCase):
    def test_visualize(self):
        seg = audiosegment.from_file("furelise.wav")[:25000]
        spec, freqs = seg.filter_bank(nfilters=5, mode='log')
        if os.environ.get('DISPLAY', False):
            visualize(spec, freqs)

    def test_no_exceptions(self):
        seg = audiosegment.from_file("furelise.wav")[:25000]
        _spec, _freqs = seg.filter_bank(nfilters=5, mode='mel')
        _spec, _freqs = seg.filter_bank(nfilters=5, mode='log')


if __name__ == "__main__":
    unittest.main()
