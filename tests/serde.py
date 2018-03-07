"""
Tests serialization and deserialization of an audiosegment.
"""
import importlib.util
__spec = importlib.util.spec_from_file_location("audiosegment", "../audiosegment.py")
asg = importlib.util.module_from_spec(__spec)
__spec.loader.exec_module(asg)
import read_from_file
import sys

def test(seg):
    print("Serializing segment...")
    ser = seg.serialize()
    print("Deserializing...")
    des = asg.deserialize(ser)
    print("Name:", des.name)
    return des

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
