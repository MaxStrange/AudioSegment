"""
Tests pickling and unpickling of an audiosegment.
"""

# Do path-style import hacking instead of importlib-style import hacking else pickling breaks like:
#   _pickle.PicklingError: Can't pickle <class 'audiosegment.AudioSegment'>: it's not the same object as audiosegment.AudioSegment
import sys
sys.path.insert(0, '../')
import audiosegment as asg
# import importlib.util
# __spec = importlib.util.spec_from_file_location("audiosegment", "../audiosegment.py")
# asg = importlib.util.module_from_spec(__spec)
# __spec.loader.exec_module(asg)
# import read_from_file

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
