"""
Tests doing a spectrogram using an AudioSegment.
"""
import numpy as np
import read_from_file
import sys
import visualize

def test(seg):
    print("Removing silence...")
    result = seg.filter_silence()
    outname_silence = "results/nosilence.wav"
    result.export(outname_silence, format="wav")
    visualize.visualize(result[:min(visualize.VIS_MS, len(result))], title="After Silence Removal")
    print("After removal:", outname_silence)

    # Now try again, but with massive threshold for silence removal
    # This will strip almost every sample in the file, leaving a practically empty
    # WAV file, which Pydub chokes on.
    _ = seg.filter_silence(threshold_percentage=99.9)

    return result

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
