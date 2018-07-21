"""
Tests Computer-aided Auditory Scene Analysis algorithm
"""
import read_from_file
import sys

def test(seg):
    # 20s of audio
    seg[:20000].auditory_scene_analysis()

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
