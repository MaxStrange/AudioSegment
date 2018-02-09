"""
Tests the ability to trim the audiosegment into slices.
"""
import read_from_file
import sys

def test(seg):
    print("Trimming to 30 ms slices...")
    slices = seg.dice(seconds=0.03, zero_pad=True)
    print("  |-> Got", len(slices), "slices.")
    print("  |-> Durations in seconds of each slice:", [sl.duration_seconds for sl in slices])
    return slices

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
