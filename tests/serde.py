"""
Tests pickling and unpickling of an audiosegment.
"""
import sys
sys.path.insert(0, '../')
import audiosegment
import numpy as np
import pickle
import unittest

class TestSerde(unittest.TestCase):
    """
    Test serialization and deserialization does not corrupt data.
    """
    def test_pack_and_unpack_pickle(self):
        seg = audiosegment.from_file("stereo_furelise.wav")
        serialized = pickle.dumps(seg)
        deserialzd = pickle.loads(serialized)

        self.assertEqual(seg.channels, deserialzd.channels)
        self.assertEqual(seg.frame_rate, deserialzd.frame_rate)
        self.assertEqual(seg.duration_seconds, deserialzd.duration_seconds)
        self.assertTrue(np.allclose(seg.to_numpy_array(), deserialzd.to_numpy_array()))

    def test_pack_and_unpack(self):
        seg = audiosegment.from_file("stereo_furelise.wav")
        serialized = seg.serialize()
        deserialzd = audiosegment.deserialize(serialized)

        self.assertEqual(seg.channels, deserialzd.channels)
        self.assertEqual(seg.frame_rate, deserialzd.frame_rate)
        self.assertEqual(seg.duration_seconds, deserialzd.duration_seconds)
        self.assertTrue(np.allclose(seg.to_numpy_array(), deserialzd.to_numpy_array()))

if __name__ == "__main__":
    unittest.main()
