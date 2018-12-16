"""
Tests Computer-aided Auditory Scene Analysis algorithm
"""
import sys
sys.path.insert(0, '../')
import audiosegment as asg
import algorithms.asa as asa
import numpy as np
import read_from_file

def _test_match_case(onset_front_id, expected_match, onset_fronts, offset_fronts, onsets, offsets, test_title):
    """
    Run one test case from unittest_front_matching().
    """
    print("EXPECTED SEGMENTATION MASK:\n", expected_match)
    outcome = asa._match_fronts(onset_fronts, offset_fronts, onsets, offsets)
    # Assert that every onset_front_id in the outcome segmentation mask has a corresponding value at the same index in
    # expected_match, and vice versa
    assert outcome.shape == expected_match.shape, "TEST {}: Shape of outcome ({}) does not match expected {}".format(
        test_title, outcome.shape, expected_match.shape
    )
    for i, row in enumerate(outcome[:, :]):
        for j, item in enumerate(row):
            if item == onset_front_id:
                assert item == expected_match[i, j], "TEST {}: Mismatch at: expected[{}, {}] = {}, outcome[{}, {}] = {}".format(
                    test_title, i, j, expected_match[i, j], i, j, outcome[i, j]
                )
    for i, row in enumerate(expected_match[:, :]):
        for j, item in enumerate(row):
            if item == onset_front_id:
                assert item == outcome[i, j], "TEST {}: Mismatch at: expected[{}, {}] = {}, outcome[{}, {}] = {};\nExpected:\n{}\nGot:\n{}".format(
                    test_title, i, j, expected_match[i, j], i, j, outcome[i, j], expected_match, outcome
                )
    print("-------- PASS TEST {}-----------".format(test_title))

def unittest_front_matching(seg):
    """
    Run a test suite for a particularly hairy portion of the CASA algorithm.
    """
    #### TEST CASE 1 & 2 & 3 ####
    onsets = np.array([
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
    ])
    onset_fronts = np.array([
        [0, 2, 0, 0, 3, 0, 0, 4, 0, 0],
        [0, 2, 0, 3, 0, 0, 4, 0, 0, 0],
        [0, 2, 0, 0, 3, 0, 4, 0, 0, 0]
    ])

    offsets = np.array([
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    ])
    offset_fronts = np.array([
        [0, 0, 0, 2, 0, 0, 3, 0, 0, 0],
        [0, 0, 2, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 3, 0, 0, 0, 0]
    ])

    expected_segmentation_mask = np.array([
        [0, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        [0, 2, 2, 3, 3, 0, 4, 4, 4, 4],
        [0, 2, 2, 0, 3, 3, 4, 4, 4, 4]
    ])
    _test_match_case(onset_front_id=2, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="1")
    _test_match_case(onset_front_id=3, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="2")
    _test_match_case(onset_front_id=4, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="3")

    #### TEST CASE 4 & 5 & 6 ####
    onsets = np.array([
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    ])
    onset_fronts = np.array([
        [2, 3, 0, 0, 0, 4, 0, 0, 0, 0],
        [2, 3, 0, 0, 0, 0, 0, 4, 0, 0],
        [2, 3, 0, 0, 0, 4, 0, 0, 0, 0]
    ])

    offsets = np.array([
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    ])
    offset_fronts = np.array([
        [0, 0, 2, 0, 3, 0, 4, 0, 5, 6],
        [0, 0, 2, 0, 0, 3, 4, 0, 5, 6],
        [0, 0, 2, 0, 3, 0, 4, 0, 5, 6]
    ])

    expected_segmentation_mask = np.array([
        [2, 2, 2, 0, 0, 4, 4, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 4, 4, 0, 0],
        [2, 2, 2, 0, 0, 4, 4, 0, 0, 0]
    ])

    _test_match_case(onset_front_id=2, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="4")
    _test_match_case(onset_front_id=3, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="5")
    _test_match_case(onset_front_id=4, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="6")

    onsets = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    ])
    onset_fronts = np.array([
        [2, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 3, 0, 0, 0, 0, 4, 0],
        [2, 0, 0, 0, 0, 3, 0, 0, 4, 0],
        [2, 0, 0, 0, 0, 3, 0, 0, 0, 0]
    ])

    offsets = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    ])
    offset_fronts = np.array([
        [0, 2, 0, 0, 3, 0, 0, 0, 5, 0],
        [0, 2, 0, 0, 3, 0, 4, 0, 5, 0],
        [0, 2, 0, 0, 3, 0, 4, 0, 0, 5],
        [0, 0, 0, 0, 3, 0, 4, 0, 0, 5],
        [0, 0, 0, 0, 3, 0, 0, 0, 5, 0]
    ])

    expected_segmentation_mask = np.array([
        [2, 2, 3, 3, 3, 0, 0, 0, 0, 0],
        [2, 2, 3, 3, 3, 0, 0, 0, 0, 0],
        [2, 2, 0, 3, 3, 0, 0, 0, 4, 4],
        [2, 2, 2, 2, 2, 3, 3, 0, 4, 4],
        [2, 2, 2, 2, 2, 3, 3, 3, 3, 0]

    ])
    _test_match_case(onset_front_id=2, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="7")
    _test_match_case(onset_front_id=3, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="8")
    _test_match_case(onset_front_id=4, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="9")

    # Test what happens when two offset fronts are tied for most offset points
    onsets = np.array([
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    ])
    onset_fronts = np.array([
        [2, 0, 0, 0, 3, 0, 0, 4, 0, 0],
        [2, 0, 0, 0, 3, 0, 0, 4, 0, 0],
        [2, 0, 0, 0, 3, 0, 0, 4, 0, 0],
        [2, 0, 0, 0, 3, 0, 0, 4, 0, 0],
        [2, 0, 0, 0, 3, 0, 0, 4, 0, 0]
    ])

    offsets = np.array([
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    ])
    offset_fronts = np.array([
        [0, 2, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 3, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 4],
        [0, 6, 0, 0, 0, 0, 5, 0, 7, 0],
        [0, 6, 0, 0, 0, 0, 0, 0, 7, 0]
    ])

    expected_segmentation_mask = np.array([
        [2, 2, 0, 0, 3, 3, 0, 4, 4, 4],
        [2, 2, 0, 0, 3, 3, 0, 4, 4, 4],
        [2, 2, 2, 2, 2, 2, 2, 4, 4, 4],
        [2, 2, 0, 0, 3, 3, 3, 4, 4, 0],
        [2, 2, 0, 0, 3, 3, 3, 3, 3, 0]
    ])

    _test_match_case(onset_front_id=2, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="10")
    _test_match_case(onset_front_id=3, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="11")
    _test_match_case(onset_front_id=4, expected_match=expected_segmentation_mask, onset_fronts=onset_fronts, offset_fronts=offset_fronts, onsets=onsets, offsets=offsets, test_title="12")

def _test_front_case(function_input, expected_output, sample_rate_hz, threshold_ms, test_name):
    """
    Test whether AudioSegment._form_onset_offset(function_input, sample_rate_hz, threshold_ms) == expected_output.
    """
    output = asa._form_onset_offset_fronts(function_input, sample_rate_hz, threshold_ms)
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

def unittest_overlapping(seg):
    """
    Tests the overlapping algorithm in the ASA module.
    """
    other    = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 0, 0, 0, 4, 4, 4, 4, 4],
                         [0, 2, 2, 0, 3, 3, 4, 4, 4, 4, 4],
                         [0, 0, 0, 0, 3, 3, 3, 0, 4, 4, 0]])

    toupdate = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 4, 0, 0],
                         [2, 2, 2, 2, 0, 0, 0, 0, 4, 4, 4],
                         [2, 2, 2, 2, 0, 3, 3, 0, 4, 4, 4],
                         [0, 0, 0, 0, 0, 3, 3, 0, 4, 4, 0]])

    # Overlaps
    expected = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 4, 0, 0],
                         [2, 2, 2, 2, 0, 0, 0, 0, 4, 4, 4],
                         [2, 2, 2, 2, 0, 3, 3, 0, 4, 4, 4],
                         [0, 0, 0, 0, 0, 3, 3, 0, 4, 4, 0]])

    output = np.copy(toupdate)
    asa._update_segmentation_mask_if_overlap(output, other, id=2, otherid=2)
    assert np.all(output == expected), "Expected\n{}\nbut got\n{}".format(expected, output)
    print("PASS TEST", (2, 2))

    expected = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 4, 0, 0],
                         [2, 2, 2, 2, 0, 0, 0, 0, 4, 4, 4],
                         [2, 2, 2, 2, 3, 3, 3, 0, 4, 4, 4],
                         [0, 0, 0, 0, 3, 3, 3, 0, 4, 4, 0]])

    output = np.copy(toupdate)
    asa._update_segmentation_mask_if_overlap(output, other, id=3, otherid=3)
    assert np.all(output == expected), "Expected\n{}\nbut got\n{}".format(expected, output)
    print("PASS TEST", (3, 3))

    expected = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 4, 0, 0],
                         [2, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3],
                         [2, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3],
                         [0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0]])

    output = np.copy(toupdate)
    asa._update_segmentation_mask_if_overlap(output, other, id=3, otherid=4)
    assert np.all(output == expected), "Expected\n{}\nbut got\n{}".format(expected, output)
    print("PASS TEST", (3, 4))

    expected = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 4, 0, 0],
                         [2, 2, 2, 2, 0, 0, 4, 4, 4, 4, 4],
                         [2, 2, 2, 2, 0, 3, 4, 4, 4, 4, 4],
                         [0, 0, 0, 0, 0, 3, 3, 0, 4, 4, 0]])

    output = np.copy(toupdate)
    asa._update_segmentation_mask_if_overlap(output, other, id=4, otherid=4)
    assert np.all(output == expected), "Expected\n{}\nbut got\n{}".format(expected, output)
    print("PASS TEST", (4, 4))

    # No overlaps
    expected = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
                         [0, 2, 2, 0, 0, 0, 0, 0, 4, 0, 0],
                         [2, 2, 2, 2, 0, 0, 0, 0, 4, 4, 4],
                         [2, 2, 2, 2, 0, 3, 3, 0, 4, 4, 4],
                         [0, 0, 0, 0, 0, 3, 3, 0, 4, 4, 0]])

    output = np.copy(toupdate)
    asa._update_segmentation_mask_if_overlap(output, other, id=2, otherid=3)
    assert np.all(output == expected), "Expected\n{}\nbut got\n{}".format(expected, output)
    print("PASS TEST", (2, 3))

    output = np.copy(toupdate)
    asa._update_segmentation_mask_if_overlap(output, other, id=2, otherid=4)
    assert np.all(output == expected), "Expected\n{}\nbut got\n{}".format(expected, output)
    print("PASS TEST", (2, 4))

def unittest_adjacent_segments(seg):
    """
    Test the algorithm that determines whether segments are adjacent.
    """
    mask = np.array([[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 2, 2, 2, 2, 0, 0, 0, 4, 0, 0],
                     [0, 2, 2, 0, 0, 0, 0, 0, 4, 0, 0],
                     [2, 2, 2, 2, 0, 0, 0, 0, 4, 4, 4],
                     [2, 2, 2, 2, 0, 3, 3, 3, 4, 4, 4],
                     [0, 0, 0, 0, 0, 3, 3, 0, 4, 4, 0]])

    seg2 = np.where(mask == 2)
    seg3 = np.where(mask == 3)
    seg4 = np.where(mask == 4)

    result = asa._segments_are_adjacent(seg2, seg3)
    assert result == False, "Segment 2 and 3 are not adjacent, but result is True"
    result = asa._segments_are_adjacent(seg3, seg2)
    assert result == False, "Segment 3 and 2 are not adjacent, but result is True"

    result = asa._segments_are_adjacent(seg2, seg4)
    assert result == False, "Segment 2 and 4 are not adjacent, but result is True"
    result = asa._segments_are_adjacent(seg4, seg2)
    assert result == False, "Segment 4 and 2 are not adjacent, but result is True"

    result = asa._segments_are_adjacent(seg3, seg4)
    assert result == True, "Segment 3 and 4 are adjacent, but result is False"
    result = asa._segments_are_adjacent(seg4, seg3)
    assert result == True, "Segment 4 and 3 are adjacent, but result is False"

def test(seg):
    wavs = seg[:20000].auditory_scene_analysis(debug=True, debugplot=False)
    ntosave = 5
    print("Got", len(wavs), "back. Saving up to the first", ntosave, "of them to disk")
    for i in range(min(ntosave, len(wavs))):
        name = "asa_results_{}.wav".format(i)
        wavs[i].export(name, format="wav")

if __name__ == "__main__":
    seg = read_from_file.test(sys.argv[1])
    unittest_adjacent_segments(seg)
    unittest_overlapping(seg)
    unittest_front_matching(seg)
    unittest_front_formation(seg)
    test(seg)
