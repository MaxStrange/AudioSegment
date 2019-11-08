"""
Tests if something is human audible.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import unittest


class TestHumanAudible(unittest.TestCase):
    """
    Test the human audible method.
    """
    def test_should_be_audible(self):
        seg = audiosegment.from_file("furelise.wav")
        seconds_audible = seg.human_audible()
        self.assertGreaterEqual(seconds_audible, seg.duration_seconds * 0.85, "This file should contain at least 85 percent audible sound but does not.")

    def test_should_not_be_audible(self):
        seg = audiosegment.silent(1000)
        seconds_audible = seg.human_audible()
        self.assertLessEqual(seconds_audible, seg.duration_seconds * 0.05, "This segment should not be audible, but is.")


if __name__ == "__main__":
    unittest.main()
