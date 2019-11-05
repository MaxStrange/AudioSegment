"""
This file tests the `generate_frames` method.
"""
import math
import numpy as np
import read_from_file
import sys
sys.path.insert(0, '../')
import audiosegment as asg

def test(before):
    nchannels = before.channels
    bps = before.sample_width
    hz = before.frame_rate
    duration_s = before.duration_seconds

    results = [s for s, _ in before.generate_frames_as_segments(1000, zero_pad=False)]
    after = results[0].reduce(results[1:])

    assert after.channels == nchannels, "Got {} channels, expected {}.".format(after.channels, nchannels)
    assert after.sample_width == bps, "Got {} sample width, expected {}.".format(after.sample_width, bps)
    assert after.frame_rate == hz, "Got {} frame rate, expected {}.".format(after.frame_rate, hz)
    assert after.duration_seconds == duration_s, "Got {} duration seconds, expected {}.".format(after.duration_seconds, duration_s)

    beforearr = before.to_numpy_array()
    afterarr = after.to_numpy_array()

    assert np.allclose(beforearr, afterarr), "Segments differ in data"

    return after

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
