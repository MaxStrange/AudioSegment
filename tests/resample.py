"""
Tests the resampling of audiosegment objects by resampling and visualizing
them in several different ways.
"""
import sys
import visualize

def test(seg):
    visualize.visualize(seg, title="Raw from WAV file")

    print("Resampling to 32kHz, mono, 16-bit...")
    seg32_m_16 = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=1)
    visualize.visualize(seg32_m_16, title="Resampled to 16 bit @ 32kHz")

    print("Resampling to 16kHz, mono, 16-bit...")
    seg16_m_16 = seg.resample(sample_rate_Hz=16000, sample_width=2, channels=1)
    visualize.visualize(seg16_m_16, title="Resampled to 16 bit @ 16kHz")

    print("Resampling to 16kHz, mono, 32-bit...")
    seg16_m_32 = seg.resample(sample_rate_Hz=16000, sample_width=4, channels=1)
    visualize.visualize(seg16_m_32, title="Resampled to 32 bit @ 16kHz")

    print("Resampling to 24kHz, mono, 24-bit...")
    seg24_m_24 = seg.resample(sample_rate_Hz=24000, sample_width=3, channels=1)
    visualize.visualize(seg24_m_24, title="Resampled to 24 bit @ 24kHz")

    print("Using 16kHz @ 16 bit")

    return seg16_m_16

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    test(seg)
