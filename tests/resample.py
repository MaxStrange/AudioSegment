"""
Tests the resampling of audiosegment objects by resampling and visualizing
them in several different ways.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import common
import numpy as np
import sys
import unittest


class TestResample(unittest.TestCase):
    """
    """
    def _compare(self, seg, hz, nchannels, nbytes):
        self.assertEqual(seg.frame_rate, hz), "Segment {} has frame rate of {}, was expecting {}".format(seg, seg.frame_rate, hz)
        self.assertEqual(seg.channels, nchannels), "Segment {} has {} channels, was expecting {}".format(seg, seg.channels, nchannels)
        self.assertEqual(seg.sample_width, nbytes), "Segment {} has sample width of {}, was expecting {}".format(seg, seg.sample_width, nbytes)

    def _check_underlying_data(self, another, seg):
        """
        Checks to make sure seg.raw_data's bytes are all equal to the values in arr.
        """
        bools = [s == v for s, v in zip(another.raw_data, seg.raw_data)]
        self.assertTrue(all(bools))

    def test_resample_hz(self):
        """
        Test that resampling does what we expect.
        """
        seg = audiosegment.from_file("furelise.wav")

        for hz in (8000, 16000, 32000, 44100, 48000, 23411, 96000):
            with self.subTest(hz):
                resampled = seg.resample(sample_rate_Hz=hz)
                self._compare(resampled, hz, seg.channels, seg.sample_width)

    def test_resample_channels(self):
        """
        Test that upmixing and downmixing does what we expect.
        """
        segmono = audiosegment.from_file("furelise.wav")
        segster = audiosegment.from_file("furelise.wav")
        seg16   = audiosegment.from_mono_audiosegments(*[segmono for _ in range(16)])

        for ch in (1, 2, 3, 4, 8, 16):
            with self.subTest(ch):
                resampled = segmono.resample(channels=ch)
                self._compare(resampled, segmono.frame_rate, ch, segmono.sample_width)

                resampled = segster.resample(channels=ch)
                self._compare(resampled, segmono.frame_rate, ch, segmono.sample_width)

                resampled = seg16.resample(channels=ch)
                self._compare(resampled, segmono.frame_rate, ch, segmono.sample_width)

    def test_resample_sample_width(self):
        """
        Test that changing the sample width does what we expect.
        """
        seg = audiosegment.from_file("furelise.wav")

        for width in (1, 2, 4):
            with self.subTest(width):
                resampled = seg.resample(sample_width=width)
                self._compare(resampled, seg.frame_rate, seg.channels, width)

    def test_upmix_then_downmix_mono(self):
        """
        Test that upmixing and then downmixing does not change the audio.
        """
        seg = audiosegment.from_file("furelise.wav")
        remixed = seg.resample(channels=8)
        unmixed = remixed.resample(channels=1)
        self._check_underlying_data(seg, unmixed)
        self.assertTrue(common.is_playable(unmixed))

    def test_upmix_then_downmix_stereo(self):
        """
        """
        seg = audiosegment.from_file("stereo_furelise.wav")
        remixed = seg.resample(channels=8)
        unmixed = remixed.resample(channels=2)
        self._check_underlying_data(seg, unmixed)
        self.assertTrue(common.is_playable(unmixed))

    def test_upmixing_does_not_change(self):
        """
        Test that upmixing just results in two identical channels.
        """
        seg = audiosegment.from_file("furelise.wav")
        remixed = seg.resample(channels=2)
        self.assertTrue(np.allclose(seg.to_numpy_array(), remixed.to_numpy_array()[:,0]))
        self.assertTrue(np.allclose(seg.to_numpy_array(), remixed.to_numpy_array()[:,1]))

    def test_downmixing_playable(self):
        """
        Test that downmixing results in playable audio.
        """
        seg = audiosegment.from_file("stereo_furelise.wav")
        mono = seg.resample(channels=1)
        self.assertTrue(common.is_playable(mono))

    def test_misc_combos(self):
        """
        Test miscellaneous combinations.
        """
        seg = audiosegment.from_file("stereo_furelise.wav")
        seg32_m_16 = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=1)
        self._compare(seg32_m_16, 32000, 1, 2)

        seg16_m_32 = seg.resample(sample_rate_Hz=16000, sample_width=4, channels=1)
        self._compare(seg16_m_32, 16000, 1, 4)

        seg24_m_24 = seg.resample(sample_rate_Hz=24000, sample_width=3, channels=1)
        self._compare(seg24_m_24, 24000, 1, 4)  # Pydub converts 3-byte to 4-byte internally

        seg16_m_16 = seg.resample(sample_rate_Hz=16000, sample_width=2, channels=1)
        self._compare(seg16_m_16, 16000, 1, 2)

        seg32_2_16 = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=2)
        self._compare(seg32_2_16, 32000, 2, 2)

        seg32_8_16 = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=8)
        self._compare(seg32_8_16, 32000, 8, 2)

        seg32_4_32 = seg32_8_16.resample(sample_width=4, channels=4)
        self._compare(seg32_4_32, 32000, 4, 4)

        seg32_8_32 = seg32_4_32.resample(channels=8)
        self._compare(seg32_8_32, 32000, 8, 4)

        seg32_2_32 = seg32_8_32.resample(channels=2)
        self._compare(seg32_2_32, 32000, 2, 4)

        seg32_1_32 = seg32_2_32.resample(channels=1)
        self._compare(seg32_1_32, 32000, 1, 4)

        seg8_15_8 = seg32_1_32.resample(sample_rate_Hz=8000, sample_width=1, channels=15)
        self._compare(seg8_15_8, 8000, 15, 1)

        seg8_11_8 = seg8_15_8.resample(channels=11)
        self._compare(seg8_11_8, 8000, 11, 1)


if __name__ == "__main__":
    unittest.main()
