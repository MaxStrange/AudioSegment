"""
Tests if something is human audible.
"""
import read_from_file
import sys

def test(seg):
    return seg.human_audible()

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    seconds_audible = test(seg)
    print("Fraction of audio that is human audible:", seconds_audible / (len(seg) / 1000))
