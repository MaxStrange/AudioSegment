"""
Tests the resampling of audiosegment objects by resampling and visualizing
them in several different ways.
"""
import read_from_file
import sys
import visualize

def _compare(seg, hz, nchannels, nbytes):
    assert seg.frame_rate == hz, "Segment {} has frame rate of {}, was expecting {}".format(seg, seg.frame_rate, hz)
    assert seg.channels == nchannels, "Segment {} has {} channels, was expecting {}".format(seg, seg.channels, nchannels)
    assert seg.sample_width == nbytes, "Segment {} has sample width of {}, was expecting {}".format(seg, seg.sample_width, nbytes)

def test(seg):
#    visualize.visualize(seg, title="Raw from WAV file")

    print("Resampling to 32kHz, mono, 16-bit...")
    seg32_m_16 = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=1)
    visualize.visualize(seg32_m_16, title="Resampled to 16 bit @ 32kHz")

    # Test a few possibilities
    seg16_m_32 = seg.resample(sample_rate_Hz=16000, sample_width=4, channels=1)
    _compare(seg16_m_32, 16000, 1, 4)

    seg24_m_24 = seg.resample(sample_rate_Hz=24000, sample_width=3, channels=1)
    _compare(seg24_m_24, 24000, 1, 4)  # Pydub converts 3-byte to 4-byte internally

    seg16_m_16 = seg.resample(sample_rate_Hz=16000, sample_width=2, channels=1)
    _compare(seg16_m_16, 16000, 1, 2)

    seg32_2_16 = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=2)
    _compare(seg32_2_16, 32000, 2, 2)

    seg32_8_16 = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=8)
    _compare(seg32_8_16, 32000, 8, 2)

    seg32_4_32 = seg32_8_16.resample(sample_width=4, channels=4)
    _compare(seg32_4_32, 32000, 4, 4)

    seg32_8_32 = seg32_4_32.resample(channels=8)
    _compare(seg32_8_32, 32000, 8, 4)

    seg32_2_32 = seg32_8_32.resample(channels=2)
    _compare(seg32_2_32, 32000, 2, 4)

    seg32_1_32 = seg32_2_32.resample(channels=1)
    _compare(seg32_1_32, 32000, 1, 4)

    seg8_15_8 = seg32_1_32.resample(sample_rate_Hz=8000, sample_width=1, channels=15)
    _compare(seg8_15_8, 8000, 15, 1)

    seg8_11_8 = seg8_15_8.resample(channels=11)
    _compare(seg8_11_8, 8000, 11, 1)

    print("Using 32kHz & 16 bit")

    return seg32_m_16

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
