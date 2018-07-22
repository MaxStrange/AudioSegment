"""
This module contains a bunch of functions that are integral to the
auditory event detection algorithm used by AudioSegment. We refactor
them to here because they aren't really useful on their own, and they
take up brainspace by being in the AudioSegment class.
"""
import random

def _get_filter_indices(seg, start_as_yes, prob_raw_yes, ms_per_input, model, transition_matrix, model_stats):
    """
    This has been broken out of the `filter` function to reduce cognitive load.
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
    This has been broken out of the `filter` function to reduce cognitive load.
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
    ls is a list of 1s or 0s for when the filter is on or off
    """
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

