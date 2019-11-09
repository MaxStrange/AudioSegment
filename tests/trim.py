"""
Tests the ability to trim the audiosegment into slices.
"""
import sys
sys.path.insert(0, '../')
import algorithms.util as util
import audiosegment
import math
import sys
import unittest


class TestDice(unittest.TestCase):
    """
    Test the dice method.
    """
    def test_dice(self):
        seg = audiosegment.from_file("furelise.wav")
        dice_len_s = 0.03

        slices = seg.dice(seconds=dice_len_s, zero_pad=True)
        for i, sl in enumerate(slices):
            msg = "Slice {} out of {} is of duration {}s, but should be of duration {}s".format(i, len(slices), sl.duration_seconds, dice_len_s)
            self.assertTrue(util.isclose(sl.duration_seconds, dice_len_s), msg)


if __name__ == "__main__":
    unittest.main()
