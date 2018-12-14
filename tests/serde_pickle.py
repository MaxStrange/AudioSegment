"""
Tests pickling and unpickling of an audiosegment.
"""

import sys
sys.path.insert(0, '../')
import audiosegment as asg

import pickle

def test(seg):
    print("Pickling segment...")
    ser = pickle.dumps(seg)
    print("Unpickling...")
    des = pickle.loads(ser)
    print("Name:", des.name)
    print("Duration:", des.duration_seconds)
    return des

if __name__ == "__main__":
    seg = asg.from_file(sys.argv[1])
    test(seg)
