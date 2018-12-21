"""
Tests the ability to trim the audiosegment into slices.
"""
import sys
sys.path.insert(0, '../')
import algorithms.util as util
import math
import read_from_file
import sys

def test(seg):
    dice_len_s = 0.03
    print("Trimming to {}s slices...".format(dice_len_s))
    slices = seg.dice(seconds=dice_len_s, zero_pad=True)
    print("  |-> Got", len(slices), "slices.")
    print("  |-> Checking each one to make sure it is {}s in length...".format(dice_len_s))
    for i, sl in enumerate(slices):
        assert util.isclose(sl.duration_seconds, dice_len_s), "Slice {} out of {} is of duration {}s, but should be of duration {}s".format(
            i, len(slices), sl.duration_seconds, dice_len_s
        )
    return slices

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
