"""
This file tests the `generate_frames` method.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import math
import numpy as np
import unittest


class TestGenerateFrames(unittest.TestCase):
    """
    Test the generate_frames_* methods.
    """
    def test_reconstruction_mono(self):
        """
        Test that we can put the original segment back together via the frames.
        """
        before = audiosegment.from_file("furelise.wav")

        nchannels = before.channels
        bps = before.sample_width
        hz = before.frame_rate
        duration_s = before.duration_seconds

        results = [s for s, _ in before.generate_frames_as_segments(1000, zero_pad=False)]
        after = results[0].reduce(results[1:])

        self.assertEqual(after.channels, nchannels, "Got {} channels, expected {}.".format(after.channels, nchannels))
        self.assertEqual(after.sample_width, bps, "Got {} sample width, expected {}.".format(after.sample_width, bps))
        self.assertEqual(after.frame_rate, hz, "Got {} frame rate, expected {}.".format(after.frame_rate, hz))
        self.assertEqual(after.duration_seconds, duration_s, "Got {} duration seconds, expected {}.".format(after.duration_seconds, duration_s))

        beforearr = before.to_numpy_array()
        afterarr = after.to_numpy_array()

        self.assertTrue(np.allclose(beforearr, afterarr), "Segments differ in data")

    def test_reconstruction_stereo(self):
        """
        """
        before = audiosegment.from_file("stereo_furelise.wav")

        nchannels = before.channels
        bps = before.sample_width
        hz = before.frame_rate
        duration_s = before.duration_seconds

        results = [s for s, _ in before.generate_frames_as_segments(1000, zero_pad=False)]
        after = results[0].reduce(results[1:])

        self.assertEqual(after.channels, nchannels, "Got {} channels, expected {}.".format(after.channels, nchannels))
        self.assertEqual(after.sample_width, bps, "Got {} sample width, expected {}.".format(after.sample_width, bps))
        self.assertEqual(after.frame_rate, hz, "Got {} frame rate, expected {}.".format(after.frame_rate, hz))
        self.assertEqual(after.duration_seconds, duration_s, "Got {} duration seconds, expected {}.".format(after.duration_seconds, duration_s))

        beforearr = before.to_numpy_array()
        afterarr = after.to_numpy_array()

        self.assertTrue(np.allclose(beforearr, afterarr), "Segments differ in data")



if __name__ == "__main__":
    unittest.main()
