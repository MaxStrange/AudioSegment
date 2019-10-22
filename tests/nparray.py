"""
"""
import read_from_file
import sys
sys.path.insert(0, '../')
import audiosegment as asg

def test(seg):
    nchannels = seg.channels
    bps = seg.sample_width
    hz = seg.frame_rate
    duration_s = seg.duration_seconds

    nparr = seg.to_numpy_array()
    seg = asg.from_numpy_array(nparr, hz)

    assert seg.channels == nchannels, "Got {} channels, expected {}.".format(seg.channels, nchannels)
    assert seg.sample_width == bps, "Got {} sample width, expected {}.".format(seg.sample_width, bps)
    assert seg.frame_rate == hz, "Got {} frame rate, expected {}.".format(seg.frame_rate, hz)
    assert seg.duration_seconds == duration_s, "Got {} duration seconds, expected {}.".format(seg.duration_seconds, duration_s)

    return seg

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
