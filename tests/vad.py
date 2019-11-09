"""
Tests the voice activity detection.
"""
import sys
sys.path.insert(0, '../')
import algorithms.util as util
import audiosegment
import math
import sys
import unittest


class TestVoiceActivityDetection(unittest.TestCase):
    def _run_test_at_given_hz(self, hz):
        """
        Test basic functionality at given sampling frequency.
        """
        seg = audiosegment.from_file("furelise.wav").resample(sample_rate_Hz=hz)
        results = seg.detect_voice(prob_detect_voice=0.5)
        voiced = [tup[1] for tup in results if tup[0] == 'v']
        unvoiced = [tup[1] for tup in results if tup[0] == 'u']

        # Now reduce to single segments
        if len(voiced) > 1:
            voiced_segment = voiced[0].reduce(voiced[1:])
        elif len(voiced) > 0:
            voiced_segment = voiced[0]
        else:
            voiced_segment = None

        if len(unvoiced) > 1:
            unvoiced_segment = unvoiced[0].reduce(unvoiced[1:])
        elif len(unvoiced) > 0:
            unvoiced_segment = unvoiced[0]
        else:
            unvoiced_segment = None

        self.assertTrue(unvoiced_segment is not None, "Furelise should be mostly unvoiced.")

        if voiced_segment is not None:
            self.assertGreater(unvoiced_segment.duration_seconds, voiced_segment.duration_seconds)

    def test_basic_functionality_8000(self):
        """
        Test for very basic functionality at 8 kHz.
        """
        self._run_test_at_given_hz(8000)

    def test_basic_functionality_16000(self):
        """
        Test for very basic functionality at 8 kHz.
        """
        self._run_test_at_given_hz(16000)

    def test_basic_functionality_32000(self):
        """
        Test for very basic functionality at 8 kHz.
        """
        self._run_test_at_given_hz(32000)

    def test_basic_functionality_48000(self):
        """
        Test for very basic functionality at 8 kHz.
        """
        self._run_test_at_given_hz(48000)



if __name__ == "__main__":
    unittest.main()
