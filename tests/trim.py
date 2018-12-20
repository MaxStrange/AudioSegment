"""
Tests the ability to trim the audiosegment into slices.
"""
import math
import read_from_file
import sys

def _isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0):
    try:
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    except AttributeError:
        # Running on older version of python, fall back to hand-rolled implementation
        if (rel_tol < 0.0) or (abs_tol < 0.0):
            raise ValueError("Tolerances must be non-negative, but are rel_tol: {} and abs_tol: {}".format(rel_tol, abs_tol))
        if math.isnan(a) or math.isnan(b):
            return False  # NaNs are never close to anything, even other NaNs
        if (a == b):
            return True
        if math.isinf(a) or math.isinf(b):
            return False  # Infinity is only close to itself, and we already handled that case
        diff = abs(a - b)
        return (diff <= rel_tol * abs(b)) or (diff <= rel_tol * abs(a)) or (diff <= abs_tol)

def test(seg):
    dice_len_s = 0.03
    print("Trimming to {}s slices...".format(dice_len_s))
    slices = seg.dice(seconds=dice_len_s, zero_pad=True)
    print("  |-> Got", len(slices), "slices.")
    print("  |-> Checking each one to make sure it is {}s in length...".format(dice_len_s))
    for i, sl in enumerate(slices):
        assert _isclose(sl.duration_seconds, dice_len_s), "Slice {} out of {} is of duration {}s, but should be of duration {}s".format(
            i, len(slices), sl.duration_seconds, dice_len_s
        )
    return slices

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
