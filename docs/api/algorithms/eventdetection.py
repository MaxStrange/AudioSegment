"""
This module contains a bunch of functions that are integral to the
auditory event detection algorithm used by AudioSegment. We refactored
them to here because they aren't really useful on their own, and they
take up brainspace by being in the AudioSegment class.
"""
import random

def _get_filter_indices(seg, start_as_yes, prob_raw_yes, ms_per_input, model, transition_matrix, model_stats):
    """
    Runs a Markov Decision Process over the given `seg` in chunks of `ms_per_input`, yielding `True` if
    this `ms_per_input` chunk has been classified as positive (1) and `False` if this chunk has been
    classified as negative (0).

    :param seg:                 The AudioSegment to apply this algorithm to.
    :param start_as_yes:        If True, the first `ms_per_input` chunk will be classified as positive.
    :param prob_raw_yes:        The raw probability of finding the event in any given independently sampled `ms_per_input`.
    :param ms_per_input:        The number of ms of AudioSegment to be fed into the model at a time.
    :param model:               The model, which must hava predict() function, which takes an AudioSegment of `ms_per_input`
                                number of ms and which outputs 1 if the audio event is detected in that input, 0 if not.
    :param transition_matrix:   An iterable of the form: [p(yes->no), p(no->yes)].
    :param model_stats:         An iterable of the form: [p(reality=1|output=1), p(reality=1|output=0)].
    :yields:                    `True` if the event has been classified in this chunk, `False` otherwise.
    """
    filter_triggered = 1 if start_as_yes else 0
    prob_raw_no = 1.0 - prob_raw_yes
    for segment, _timestamp in seg.generate_frames_as_segments(ms_per_input):
        yield filter_triggered
        observation = int(round(model.predict(segment)))
        assert observation == 1 or observation == 0, "The given model did not output a 1 or a 0, output: "\
                + str(observation)
        prob_hyp_yes_given_last_hyp = 1.0 - transition_matrix[0] if filter_triggered else transition_matrix[1]
        prob_hyp_no_given_last_hyp  = transition_matrix[0] if filter_triggered else 1.0 - transition_matrix[1]
        prob_hyp_yes_given_data = model_stats[0] if observation == 1 else model_stats[1]
        prob_hyp_no_given_data = 1.0 - model_stats[0] if observation == 1 else 1.0 - model_stats[1]
        hypothesis_yes = prob_raw_yes * prob_hyp_yes_given_last_hyp * prob_hyp_yes_given_data
        hypothesis_no  = prob_raw_no * prob_hyp_no_given_last_hyp  * prob_hyp_no_given_data
        # make a list of ints - each is 0 or 1. The number of 1s is hypotheis_yes * 100
        # the number of 0s is hypothesis_no * 100
        distribution = [1 for i in range(int(round(hypothesis_yes * 100)))]
        distribution.extend([0 for i in range(int(round(hypothesis_no * 100)))])
        # shuffle
        random.shuffle(distribution)
        filter_triggered = random.choice(distribution)

def _group_filter_values(seg, filter_indices, ms_per_input):
    """
    Takes a list of 1s and 0s and returns a list of tuples of the form:
    ['y/n', timestamp].
    """
    ret = []
    for filter_value, (_segment, timestamp) in zip(filter_indices, seg.generate_frames_as_segments(ms_per_input)):
        if filter_value == 1:
            if len(ret) > 0 and ret[-1][0] == 'n':
                ret.append(['y', timestamp])  # The last one was different, so we create a new one
            elif len(ret) > 0 and ret[-1][0] == 'y':
                ret[-1][1] = timestamp  # The last one was the same as this one, so just update the timestamp
            else:
                ret.append(['y', timestamp])  # This is the first one
        else:
            if len(ret) > 0 and ret[-1][0] == 'n':
                ret[-1][1] = timestamp
            elif len(ret) > 0 and ret[-1][0] == 'y':
                ret.append(['n', timestamp])
            else:
                ret.append(['n', timestamp])
    return ret

def _homogeneity_filter(ls, window_size):
    """
    Takes `ls` (a list of 1s and 0s) and smoothes it so that adjacent values are more likely
    to be the same.

    :param ls:          A list of 1s and 0s to smooth.
    :param window_size: How large the smoothing kernel is.
    :returns:           A list of 1s and 0s, but smoother.
    """
    # TODO: This is fine way to do this, but it seems like it might be faster and better to do a Gaussian convolution followed by rounding
    k = window_size
    i = k
    while i <= len(ls) - k:
        # Get a window of k items
        window = [ls[i + j] for j in range(k)]
        # Change the items in the window to be more like the mode of that window
        mode = 1 if sum(window) >= k / 2 else 0
        for j in range(k):
            ls[i+j] = mode
        i += k
    return ls
