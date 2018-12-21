"""
Tests the normalization algorithm.
"""
import read_from_file
import sys
import visualize

def test(seg):
    print("Normalizing to 40dB SPL...")
    print("  SPL before:", seg.spl, "dB")
    res = seg.normalize_spl_by_average(db=40)
    print("  SPL after:", res.spl, "dB")
    visualize.visualize(res, "After normalization to 40dB")
    return res

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
