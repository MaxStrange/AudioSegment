"""
Tests the normalization algorithm.
"""
import sys
import visualize

def test(seg):
    print("Normalizing to 40dB SPL...")
    res = seg.normalize_spl_by_average(db=40)
    visualize.visualize(res, "After normalization to 40dB")
    return res

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
