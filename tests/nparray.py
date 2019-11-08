"""
Test converting to and from numpy arrays. Also check from_mono_audiosegments function.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import common
import numpy as np
import unittest


class TestNumpyArray(unittest.TestCase):
    """
    """
    def _look_up_sample_width(self, dtype) -> int:
        """
        """
        if dtype == np.int8:
            return 1
        elif dtype == np.int16:
            return 2
        elif dtype == np.int32:
            return 4
        else:
            raise ValueError("Cannot use dtype {}".format(dtype))

    def _check_underlying_data(self, seg: audiosegment.AudioSegment, arr: np.ndarray):
        """
        Checks to make sure seg.raw_data's bytes are all equal to the values in arr.
        """
        ints = np.frombuffer(seg.raw_data, dtype=arr.dtype)
        bools = [s == v for s, v in zip(ints, arr)]
        self.assertTrue(all(bools))

    def test_mono_file_to_nparray(self):
        """
        Test that a mono file converts to a numpy array with the right data type,
        length, and underlying data.
        """
        seg = audiosegment.from_file("furelise.wav")

        for width in (1, 2, 4):
            with self.subTest(width):
                seg = seg.resample(sample_width=width)
                arr = seg.to_numpy_array()
                nsamples = int(round(seg.frame_rate * seg.duration_seconds))

                self.assertEqual(seg.sample_width, self._look_up_sample_width(arr.dtype))
                self.assertEqual(arr.shape, (nsamples,))
                self._check_underlying_data(seg, arr)

    def test_mono_to_and_from(self):
        """
        Test that a mono file converts to a numpy array and back again without any change.
        """
        seg = audiosegment.from_file("furelise.wav")

        for width in (1, 2, 4):
            with self.subTest(width):
                seg = seg.resample(sample_width=width)
                arr = seg.to_numpy_array()
                seg = audiosegment.from_numpy_array(arr, seg.frame_rate)
                nsamples = int(round(seg.frame_rate * seg.duration_seconds))

                self.assertEqual(seg.sample_width, self._look_up_sample_width(arr.dtype))
                self.assertEqual(arr.shape, (nsamples,))
                self._check_underlying_data(seg, arr)
                self.assertTrue(common.is_playable(seg))

    def test_mono_from_numpy_array(self):
        """
        Test that creating a mono audio segment from a numpy array creates
        what we expected.
        """
        duration_s = 3.5
        fs = 32000
        ftone = 4000
        arr = np.int16(100 * common.synthesize_pure_tone_array(duration_s, fs, ftone))
        seg = audiosegment.from_numpy_array(arr, fs)

        sample_width = self._look_up_sample_width(arr.dtype)
        nsamples = int(round(seg.frame_rate * seg.duration_seconds))

        self.assertEqual(seg.sample_width, sample_width)
        self.assertEqual(nsamples, len(arr))
        self.assertEqual(arr.shape, (nsamples,))
        self._check_underlying_data(seg, arr)
        self.assertTrue(common.is_playable(seg))

    def test_stereo_to_numpy_array(self):
        """
        Test that the numpy representation of a stereo file is what we expect.
        """
        seg = audiosegment.from_file("stereo_furelise.wav")
        arr = seg.to_numpy_array()

        nsamples = int(round(seg.frame_rate * seg.duration_seconds))

        self.assertEqual(seg.sample_width, self._look_up_sample_width(arr.dtype))
        self.assertEqual(arr.shape, (nsamples, 2))
        self.assertTrue(np.allclose(arr[:,0], arr[:,1]))

    def test_stereo_from_numpy_array(self):
        """
        Test that we can create and play a stereo numpy array.
        """
        duration_s = 2.0
        fs = 16000
        tone_one = 100 * common.synthesize_pure_tone_array(duration_s, fs, ft=3200)
        tone_two = 100 * common.synthesize_pure_tone_array(duration_s, fs, ft=2800)
        stereo_arr = np.array([tone_one, tone_two], dtype=np.int16).reshape((-1, 2))
        stereo_seg = audiosegment.from_numpy_array(stereo_arr, fs)
        self.assertTrue(common.is_playable(stereo_seg))

    def test_stereo_to_and_from_numpy_array(self):
        """
        Tests that we can convert a stereo file to a numpy array and then back again
        without any changes.
        """
        before = audiosegment.from_file("stereo_furelise.wav")
        arr = before.to_numpy_array()
        after = audiosegment.from_numpy_array(arr, before.frame_rate)

        self.assertEqual(before.sample_width, after.sample_width)
        self.assertEqual(before.duration_seconds, after.duration_seconds)
        self.assertEqual(before.channels, after.channels)
        self.assertSequenceEqual(before.raw_data, after.raw_data)
        self.assertTrue(common.is_playable(after))

    def _test_create_file_from_n_segments(self, mono: audiosegment.AudioSegment, nchannels: int):
        """
        Create a single segment and test it against expected, from multiple segments.
        """
        arr = mono.to_numpy_array()
        arr_multi = np.tile(arr, (nchannels, 1)).T
        multi = audiosegment.from_numpy_array(arr_multi, mono.frame_rate)

        self.assertEqual(multi.channels, nchannels)
        self.assertEqual(multi.duration_seconds, mono.duration_seconds)
        self.assertEqual(multi.frame_rate, mono.frame_rate)

        return multi

    def test_create_file_from_two_monos(self):
        """
        Tests that we can create a playable wav file from copying a single
        mono wave file into stereo.
        """
        mono = audiosegment.from_file("furelise.wav")
        multi = self._test_create_file_from_n_segments(mono, 2)
        self.assertTrue(common.is_playable(multi))

        # Now test that both channels are identical
        arr = multi.to_numpy_array()
        self.assertTrue(np.allclose(arr[:,0], arr[:,1]))

    def test_create_file_from_four_monos(self):
        """
        """
        mono = audiosegment.from_file("furelise.wav")
        multi = self._test_create_file_from_n_segments(mono, 4)
        self.assertTrue(common.is_playable(multi))

        # Now test that all channels are identical
        arr = multi.to_numpy_array()
        self.assertTrue(np.allclose(arr[:,0], arr[:,1]))
        self.assertTrue(np.allclose(arr[:,1], arr[:,2]))
        self.assertTrue(np.allclose(arr[:,2], arr[:,3]))

    def test_create_file_from_eight_monos(self):
        """
        """
        mono = audiosegment.from_file("furelise.wav")
        multi = self._test_create_file_from_n_segments(mono, 8)

        # Now test that all channels are identical
        arr = multi.to_numpy_array()
        self.assertTrue(np.allclose(arr[:,0], arr[:,1]))
        self.assertTrue(np.allclose(arr[:,1], arr[:,2]))
        self.assertTrue(np.allclose(arr[:,2], arr[:,3]))
        self.assertTrue(np.allclose(arr[:,3], arr[:,4]))
        self.assertTrue(np.allclose(arr[:,4], arr[:,5]))
        self.assertTrue(np.allclose(arr[:,5], arr[:,6]))
        self.assertTrue(np.allclose(arr[:,6], arr[:,7]))


if __name__ == "__main__":
    unittest.main()
