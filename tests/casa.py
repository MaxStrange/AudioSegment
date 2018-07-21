"""
Tests Computer-aided Auditory Scene Analysis algorithm
"""
import importlib.util
__spec = importlib.util.spec_from_file_location("audiosegment", "../audiosegment.py")
asg = importlib.util.module_from_spec(__spec)
__spec.loader.exec_module(asg)
import numpy as np
import read_from_file
import sys

def _test_front_case(function_input, expected_output, sample_rate_hz, threshold_ms, test_name):
    """
    Test whether AudioSegment._form_onset_offset(function_input, sample_rate_hz, threshold_ms) == expected_output.
    """
    output = asg.AudioSegment._form_onset_offset_fronts(function_input, sample_rate_hz, threshold_ms)
    assert np.array_equal(output, expected_output), "\n{}\n!=\n{}\n\nFAILED TEST: {}".format(output, expected_output, test_name)
    print("-------- PASS TEST {}-----------".format(test_name))

def unittest_front_formation(seg):
    """
    Run a small test suite for a particularly hairy portion of the CASA algorithm.
    """
    sample_period_s = 5E-3  # Each sample is 5 ms apart
    sample_rate_hz = 1/sample_period_s
    threshold_ms = 20
 
    #### SIMPLEST POSSIBLE CASE ####
    finput = np.array([[0, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0]])
    expected = np.array([[0, 0, 0, 2, 0, 0],
                         [0, 2, 0, 0, 0, 0],
                         [2, 0, 0, 0, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 1")

    #### TEST CASE: OUTSIDE OF TIME ####
    finput = np.array([[0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0]])
    expected = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 2")

    #### TEST CASE: ACROSS TOO FEW FREQUENCIES ####
    finput = np.array([[0, 0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0]])
    expected = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 3")

    #### TEST CASE: SLIGHTLY MORE COMPLICATED INPUT ####
    finput = np.array([[0, 0, 0, 1, 1, 1],
                       [0, 1, 0, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0]])
    expected = np.array([[0, 0, 0, 2, 0, 0],
                         [0, 2, 0, 0, 0, 0],
                         [2, 0, 0, 0, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 4")

    #### TEST CASE: MORE COMPLICATED - SHOULD HAVE TWO IDS ####
    finput = np.array([[0, 0, 0, 1, 1, 1],
                       [0, 1, 0, 0, 1, 0],
                       [1, 0, 1, 0, 0, 0]])
    expected = np.array([[0, 0, 0, 2, 3, 0],
                         [0, 2, 0, 0, 3, 0],
                         [2, 0, 3, 0, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 5")

    #### TEST CASE: SHOULD HAVE THREE IDS ACROSS THE WHOLE THING ####
    finput = np.array([[0, 0, 0, 1, 1, 0],
                       [0, 1, 0, 0, 1, 1],
                       [1, 0, 1, 0, 1, 1],
                       [0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 1, 0, 1]])
    expected = np.array([[0, 0, 0, 2, 3, 0],
                         [0, 2, 0, 0, 3, 4],
                         [2, 0, 3, 0, 4, 0],
                         [0, 0, 2, 0, 3, 4],
                         [0, 0, 0, 2, 0, 3]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 6")

    #### TEST CASE: ANOTHER ONE ####
    finput = np.array([[1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1]])
    expected = np.array([[2, 3, 4, 5, 6, 7],
                         [2, 3, 4, 5, 6, 7],
                         [2, 3, 4, 5, 6, 7],
                         [2, 3, 4, 5, 6, 7],
                         [2, 3, 4, 5, 6, 7]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 7")

    #### TEST CASE: ANOTHER ONE ####
    finput = np.array([[0, 1, 1, 1, 1, 1],
                       [1, 0, 1, 1, 1, 1],
                       [1, 1, 0, 1, 1, 1],
                       [1, 1, 1, 0, 1, 1],
                       [1, 1, 1, 1, 0, 1]])
    expected = np.array([[0, 2, 3, 4, 5, 6],
                         [2, 0, 3, 4, 5, 6],
                         [2, 3, 0, 4, 5, 6],
                         [2, 3, 4, 0, 5, 6],
                         [2, 3, 4, 5, 0, 6]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 8")

    #### TEST CASE: ANOTHER ONE ####
    finput = np.array([[0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 1, 1],
                       [1, 1, 1, 0, 1, 1],
                       [1, 1, 0, 1, 0, 0]])
    expected = np.array([[0, 2, 0, 3, 0, 4],
                         [2, 0, 3, 4, 5, 0],
                         [2, 3, 0, 4, 5, 0],
                         [2, 3, 4, 0, 5, 0],
                         [2, 3, 0, 4, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 9")

    #### TEST CASE: ANOTHER ONE ####
    finput = np.array([[0, 1, 0, 1, 0, 1],
                       [1, 0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 1, 1],
                       [1, 1, 1, 0, 1, 1],
                       [1, 1, 1, 1, 1, 1]])
    expected = np.array([[0, 2, 0, 3, 0, 4],
                         [2, 0, 3, 4, 5, 0],
                         [2, 3, 0, 4, 5, 6],
                         [2, 3, 4, 0, 5, 6],
                         [2, 3, 4, 5, 6, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 10")

    #### TEST CASE: ANOTHER ONE ####
    finput = np.array([[1, 1, 1, 1, 1, 1],
                       [1, 0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 1, 1],
                       [1, 1, 1, 0, 1, 1],
                       [1, 1, 0, 1, 0, 0]])
    expected = np.array([[2, 3, 4, 5, 0, 0],
                         [2, 0, 3, 4, 5, 0],
                         [2, 3, 0, 4, 5, 0],
                         [2, 3, 4, 0, 5, 0],
                         [2, 3, 0, 4, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 11")

    #### TEST CASE: ANOTHER ONE ####
    finput = np.array([[1, 1, 1, 1, 1, 1],
                       [1, 0, 1, 1, 1, 0],
                       [1, 1, 0, 1, 1, 1],
                       [1, 1, 1, 0, 1, 1],
                       [1, 1, 0, 1, 0, 0]])
    expected = np.array([[2, 3, 4, 5, 0, 0],
                         [2, 0, 3, 4, 5, 0],
                         [2, 3, 0, 4, 5, 0],
                         [2, 3, 4, 0, 5, 0],
                         [2, 3, 0, 4, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms, "test 12")

    #### TEST CASE: SHORTER THRESHOLDS ####
    finput = np.array([[0, 0, 0, 0, 1, 1],
                       [1, 1, 0, 0, 1, 0],
                       [1, 0, 0, 1, 1, 1],
                       [0, 1, 0, 0, 1, 0],
                       [0, 1, 1, 1, 0, 1]])
    expected = np.array([[0, 0, 0, 0, 2, 0],
                         [3, 0, 0, 0, 2, 0],
                         [3, 0, 0, 2, 0, 0],
                         [0, 3, 0, 0, 2, 0],
                         [0, 3, 0, 2, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms=5, test_name="test 13")

    #### TEST CASE: SHORTER THRESHOLDS ####
    finput = np.array([[0, 1, 0, 0, 1, 1],
                       [1, 1, 0, 0, 1, 0],
                       [1, 0, 0, 1, 1, 1],
                       [0, 1, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0, 1]])
    expected = np.array([[0, 2, 0, 0, 3, 0],
                         [2, 0, 0, 0, 3, 0],
                         [2, 0, 0, 3, 0, 0],
                         [0, 2, 0, 0, 3, 0],
                         [0, 2, 0, 3, 0, 0]])
    _test_front_case(finput, expected, sample_rate_hz, threshold_ms=5, test_name="test 14")

def test(seg):
    # 20s of audio
    seg[:20000].auditory_scene_analysis()

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    unittest_front_formation(seg)
    #test(seg)
