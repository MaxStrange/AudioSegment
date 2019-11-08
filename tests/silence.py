"""
Tests doing a spectrogram using an AudioSegment.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import numpy as np
import pickle
import unittest


class TestSilence(unittest.TestCase):
    """
    Test removing silence.
    """
    def test_silence_removal(self):
        """
        Basic test for exceptions.
        """
        seg = audiosegment.from_file("furelise.wav")
        s = seg.filter_silence()
        self.assertEqual(seg.channels, s.channels)
        self.assertEqual(seg.frame_rate, s.frame_rate)
        self.assertLess(s.duration_seconds, seg.duration_seconds)

        # Now try again, but with massive threshold for silence removal
        # This will strip almost every sample in the file, leaving a practically empty
        # WAV file, which Pydub chokes on.
        _ = seg.filter_silence(threshold_percentage=99.9)


if __name__ == "__main__":
    unittest.main()
