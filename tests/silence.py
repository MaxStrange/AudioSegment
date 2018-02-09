"""
Tests doing a spectrogram using an AudioSegment.
"""
import numpy as np
import read_from_file
import sys
import visualize

def test(seg):
    print("Removing silence...")
    seg = seg.filter_silence()
    outname_silence = "results/nosilence.wav"
    seg.export(outname_silence, format="wav")
    visualize.visualize(seg[:min(visualize.VIS_MS, len(seg))], title="After Silence Removal")
    print("After removal:", outname_silence)
    return seg

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
