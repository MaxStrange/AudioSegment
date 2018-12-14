"""
This test simply tries to construct an audiosegment object
from a filepath.
"""
import sys
sys.path.insert(0, '../')
import audiosegment as asg

def test(fp):
    print("Reading in the wave file at", fp)
    seg = asg.from_file(fp)
    return seg


if __name__ == "__main__":
    test(sys.argv[1])
