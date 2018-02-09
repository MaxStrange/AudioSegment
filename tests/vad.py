"""
Tests the voice activity detection.
"""
import sys
import visualize

def test(seg):
    results = seg.detect_voice(prob_detect_voice=0.7)
    voiced = [tup[1] for tup in results if tup[0] == 'v']
    unvoiced = [tup[1] for tup in results if tup[0] == 'u']
    print("  |-> reducing voiced segments to a single wav file 'results/voiced.wav'")
    if len(voiced) > 1:
        voiced_segment = voiced[0].reduce(voiced[1:])
    elif len(voiced) > 0:
        voiced_segment = voiced[0]
    else:
        voiced_segment = None
    if voiced_segment is not None:
        voiced_segment.export("results/voiced.wav", format="WAV")
        visualize.visualize(voiced_segment[:min(visualize.VIS_MS, len(voiced_segment))], title="Voiced Segment")

    print("  |-> reducing unvoiced segments to a single wav file 'results/unvoiced.wav'")
    if len(unvoiced) > 1:
        unvoiced_segment = unvoiced[0].reduce(unvoiced[1:])
    elif len(unvoiced) > 0:
        unvoiced_segment = unvoiced[0]
    else:
        unvoiced_segment = None
    if unvoiced_segment is not None:
        unvoiced_segment.export("results/unvoiced.wav", format="WAV")
        visualize.visualize(unvoiced_segment[:min(visualize.VIS_MS, len(unvoiced_segment))], title="Unvoiced Segment")

    return voiced_segment, unvoiced_segment

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
