"""
This test simply tries to construct an audiosegment object
from a filepath.
"""
import importlib.util
__spec = importlib.util.spec_from_file_location("audiosegment", "../audiosegment.py")
asg = importlib.util.module_from_spec(__spec)
__spec.loader.exec_module(asg)
import sys

def test(fp):
    print("Reading in the wave file at", fp)
    seg = asg.from_file(fp)
    return seg


if __name__ == "__main__":
    test(sys.argv[1])
