"""
This module extracts out a bunch of the Auditory Scene Analysis (ASA)
logic, which has grown to be a little unwieldy in the AudioSegment class.
"""
import numpy as np
import scipy.signal as signal
import sys

def visualize_time_domain(seg, title=""):
    import matplotlib.pyplot as plt
    plt.plot(seg)
    plt.title(title)
    plt.show()
    plt.clf()

def visualize(spect, frequencies, title=""):
    import matplotlib.pyplot as plt
    i = 0
    for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
        plt.subplot(spect.shape[0], 1, index + 1)
        if i == 0:
            plt.title(title)
            i += 1
        plt.ylabel("{0:.0f}".format(freq))
        plt.plot(row)
    plt.show()
    plt.clf()

def visualize_peaks_and_valleys(peaks, valleys, spect, frequencies):
    import matplotlib.pyplot as plt
    i = 0
    # Reverse everything to make it have the low fs at the bottom of the figure
    peaks = peaks[::-1, :]
    valleys = valleys[::-1, :]
    for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
        plt.subplot(spect.shape[0], 1, index + 1)
        if i == 0:
            plt.title("Peaks (red) and Valleys (blue)")
            i +=1
        plt.ylabel("{0:.0f}".format(freq))
        plt.plot(row)
        these_peaks = peaks[index]
        peak_values = these_peaks * row  # Mask off anything that isn't a peak
        these_valleys = valleys[index]
        valley_values = these_valleys * row  # Mask off anything that isn't a valley
        plt.plot(peak_values, 'ro')
        plt.plot(valley_values, 'bo')
    plt.show()
    plt.clf()

def visualize_fronts(onsets, offsets, spect, frequencies):
    import matplotlib.pyplot as plt
    i = 0
    # Reverse everything to make it have the low fs at the bottom of the figure
    onsets = onsets[::-1, :]
    offsets = offsets[::-1, :]
    for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
        plt.subplot(spect.shape[0], 1, index + 1)
        if i == 0:
            plt.title("Onsets (dotted) and Offsets (solid)")
            i +=1
        plt.ylabel("{0:.0f}".format(freq))
        plt.plot(row)
        # Cycle through all the different onsets and offsets and plot them each
        colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
        nonzero_indexes_onsets = np.reshape(np.where(onsets[index, :] != 0), (-1,))
        for x in nonzero_indexes_onsets:
            id = int(onsets[index][x])
            plt.axvline(x=x, color=colors[id % len(colors)], linestyle='--')
        nonzero_indexes_offsets = np.reshape(np.where(offsets[index, :] != 0), (-1,))
        for x in nonzero_indexes_offsets:
            id = int(offsets[index][x])
            plt.axvline(x=x, color=colors[id % len(colors)], linestyle='-')
    plt.show()
    plt.clf()

def _compute_peaks_or_valleys_of_first_derivative(s, do_peaks=True):
    """
    Takes a spectrogram and returns a 2D array of the form:

    0 0 0 1 0 0 1 0 0 0 1   <-- Frequency 0
    0 0 1 0 0 0 0 0 0 1 0   <-- Frequency 1
    0 0 0 0 0 0 1 0 1 0 0   <-- Frequency 2
    *** Time axis *******

    Where a 1 means that the value in that time bin in the spectrogram corresponds to
    a peak/valley in the first derivative.

    This function is used as part of the ASA algorithm and is not meant to be used publicly.
    """
    # Get the first derivative of each frequency in the time domain
    gradient = np.nan_to_num(np.apply_along_axis(np.gradient, 1, s), copy=False)

    # Calculate the value we will use for determinig whether something is an event or not
    threshold = np.squeeze(np.nanmean(gradient, axis=1) + np.nanstd(gradient, axis=1))

    # Look for relative extrema along the time dimension
    half_window = 4
    if do_peaks:
        indexes = [signal.argrelextrema(gradient[i, :], np.greater, order=half_window)[0] for i in range(gradient.shape[0])]
    else:
        indexes = [signal.argrelextrema(gradient[i, :], np.less, order=half_window)[0] for i in range(gradient.shape[0])]

    # indexes should now contain the indexes of possible extrema
    # But we need to filter out values that are not large enough, and we want the end result
    # to be a 1 or 0 mask corresponding to locations of extrema
    extrema = np.zeros(s.shape)
    for row_index, index_array in enumerate(indexes):
        # Each index_array is a list of indexes corresponding to all the extrema in a given row
        for col_index in index_array:
            if do_peaks and (gradient[row_index, col_index] > threshold[row_index]):
                extrema[row_index, col_index] = 1
            elif not do_peaks:
                # Note that we do not remove under-threshold values from the offsets - these will be taken care of later in the algo
                extrema[row_index, col_index] = 1
    return extrema, gradient

def _correlate_onsets_and_offsets(onsets, offsets, gradients):
    """
    Takes an array of onsets and an array of offsets, of the shape [nfrequencies, nsamples], where
    each item in these arrays is either a 0 (not an on/offset) or a 1 (a possible on/offset).

    This function returns a new offsets array, where there is a one-to-one correlation between
    onsets and offsets, such that each onset has exactly one offset that occurs after it in
    the time domain (the second dimension of the array).

    The gradients array is used to decide which offset to use in the case of multiple possibilities.
    """
    # For each freq channel:
    for freq_index, (ons, offs) in enumerate(zip(onsets[:, :], offsets[:, :])):
        # Scan along onsets[f, :] until we find the first 1
        indexes_of_all_ones = np.reshape(np.where(ons == 1), (-1,))

        # Zero out anything in the offsets up to (and including) this point
        # since we can't have an offset before the first onset
        last_idx = indexes_of_all_ones[0]
        offs[0:last_idx + 1] = 0

        if len(indexes_of_all_ones > 1):
            # Do the rest of this only if we have more than one onset in this frequency band
            for next_idx in indexes_of_all_ones[1:]:
                # Get all the indexes of possible offsets from onset index to next onset index
                offset_choices = offs[last_idx:next_idx]
                offset_choice_indexes = np.where(offset_choices == 1)

                # Assert that there is at least one offset choice
                assert np.any(offset_choices), "Offsets from {} to {} only include zeros".format(last_idx, next_idx)

                # If we have more than one choice, the offset index is the one that corresponds to the most negative gradient value
                # Convert the offset_choice_indexes to indexes in the whole offset array, rather than just the offset_choices array
                offset_choice_indexes = np.reshape(last_idx + offset_choice_indexes, (-1,))
                assert np.all(offsets[freq_index, offset_choice_indexes])
                gradient_values = gradients[freq_index, offset_choice_indexes]
                index_of_largest_from_gradient_values = np.where(gradient_values == np.min(gradient_values))[0]
                index_of_largest_offset_choice = offset_choice_indexes[index_of_largest_from_gradient_values]
                assert offsets[freq_index, index_of_largest_offset_choice] == 1

                # Zero the others
                offsets[freq_index, offset_choice_indexes] = 0
                offsets[freq_index, index_of_largest_offset_choice] = 1
                last_idx = next_idx
        else:
            # We only have one onset in this frequency band, so the offset will be the very last sample
            offsets[freq_index, :] = 0
            offsets[freq_index, -1] = 1
    return offsets

def _form_onset_offset_fronts(ons_or_offs, sample_rate_hz, threshold_ms=20):
    """
    Takes an array of onsets or offsets (shape = [nfrequencies, nsamples], where a 1 corresponds to an on/offset,
    and samples are 0 otherwise), and returns a new array of the same shape, where each 1 has been replaced by
    either a 0, if the on/offset has been discarded, or a non-zero positive integer, such that
    each front within the array has a unique ID - for example, all 2s in the array will be the front for on/offset
    front 2, and all the 15s will be the front for on/offset front 15, etc.

    Due to implementation details, there will be no 1 IDs.
    """
    threshold_s = threshold_ms / 1000
    threshold_samples = sample_rate_hz * threshold_s

    ons_or_offs = np.copy(ons_or_offs)

    claimed = []
    this_id = 2
    # For each frequency,
    for frequency_index, row in enumerate(ons_or_offs[:, :]):
        ones = np.reshape(np.where(row == 1), (-1,))

        # for each 1 in that frequency,
        for top_level_frequency_one_index in ones:
            claimed.append((frequency_index, top_level_frequency_one_index))

            found_a_front = False
            # for each frequencies[i:],
            for other_frequency_index, other_row in enumerate(ons_or_offs[frequency_index + 1:, :], start=frequency_index + 1):

                # for each non-claimed 1 which is less than theshold_ms away in time,
                upper_limit_index = top_level_frequency_one_index + threshold_samples
                lower_limit_index = top_level_frequency_one_index - threshold_samples
                other_ones = np.reshape(np.where(other_row == 1), (-1,))  # Get the indexes of all the 1s in row
                tmp = np.reshape(np.where((other_ones >= lower_limit_index)  # Get the indexes in the other_ones array of all items in bounds
                                        & (other_ones <= upper_limit_index)), (-1,))
                other_ones = other_ones[tmp]  # Get the indexes of all the 1s in the row that are in bounds
                if len(other_ones) > 0:
                    unclaimed_idx = other_ones[0]  # Take the first one
                    claimed.append((other_frequency_index, unclaimed_idx))
                elif len(claimed) < 3:
                    # revert the top-most 1 to 0
                    ons_or_offs[frequency_index, top_level_frequency_one_index] = 0
                    claimed = []
                    break  # Break from the for-each-frequencies[i:] loop so we can move on to the next item in the top-most freq
                elif len(claimed) >= 3:
                    found_a_front = True
                    # this group of so-far-claimed forms a front
                    claimed_as_indexes = tuple(np.array(claimed).T)
                    ons_or_offs[claimed_as_indexes] = this_id
                    this_id += 1
                    claimed = []
                    break  # Move on to the next item in the top-most array
            # If we never found a frequency that did not have a matching offset, handle that case here
            if len(claimed) >= 3:
                claimed_as_indexes = tuple(np.array(claimed).T)
                ons_or_offs[claimed_as_indexes] = this_id
                this_id += 1
                claimed = []
            elif found_a_front:
                this_id += 1
            else:
                ons_or_offs[frequency_index, top_level_frequency_one_index] = 0
                claimed = []

    return ons_or_offs

def _lookup_offset_by_onset_idx(onset_idx, onsets, offsets):
    """
    Takes an onset index (freq, sample) and returns the offset index (freq, sample)
    such that frequency index is the same, and sample index is the minimum of all
    offsets ocurring after the given onset. If there are no offsets after the given
    onset in that frequency channel, the final sample in that channel is returned.
    """
    assert len(onset_idx) == 2, "Onset_idx must be a tuple of the form (freq_idx, sample_idx)"
    frequency_idx, sample_idx = onset_idx
    offset_sample_idxs = np.reshape(np.where(offsets[frequency_idx, :] == 1), (-1,))
    # get the offsets which occur after onset
    offset_sample_idxs = offset_sample_idxs[offset_sample_idxs > sample_idx]
    if len(offset_sample_idxs) == 0:
        # There is no offset in this frequency that occurs after the onset, just return the last sample
        chosen_offset_sample_idx = offsets.shape[1] - 1
        assert offsets[frequency_idx, chosen_offset_sample_idx] == 0
    else:
        # Return the closest offset to the onset
        chosen_offset_sample_idx = offset_sample_idxs[0]
        assert offsets[frequency_idx, chosen_offset_sample_idx] != 0
    return frequency_idx, chosen_offset_sample_idx

def _get_front_idxs_from_id(fronts, id):
    """
    Return a list of tuples of the form (frequency_idx, sample_idx),
    corresponding to all the indexes of the given front.
    """
    freq_idxs, sample_idxs = np.where(fronts == id)
    return [(f, i) for f, i in zip(freq_idxs, sample_idxs)]

def _choose_front_id_from_candidates(candidate_offset_front_ids, offset_fronts, offsets_corresponding_to_onsets):
    """
    Returns a front ID which is the id of the offset front that contains the most overlap
    with offsets that correspond to the given onset front ID.
    """
    noverlaps = []  # will contain tuples of the form (number_overlapping, offset_front_id)
    for offset_front_id in candidate_offset_front_ids:
        offset_front_f_idxs, offset_front_s_idxs = np.where(offset_fronts == offset_front_id)
        offset_front_idxs = [(f, i) for f, i in zip(offset_front_f_idxs, offset_front_s_idxs)]
        noverlap_this_id = len(set(offset_front_idxs).symmetric_difference(set(offsets_corresponding_to_onsets)))
        noverlaps.append((noverlap_this_id, offset_front_id))
    _overlapped, chosen_offset_front_id = max(noverlaps, key=lambda t: t[0])
    return int(chosen_offset_front_id)

def _get_offset_front_id_after_onset_sample_idx(onset_sample_idx, offset_fronts):
    """
    Returns the offset_front_id which corresponds to the offset front which occurs
    first entirely after the given onset sample_idx.
    """
    # get all the offset_front_ids
    offset_front_ids = [i for i in np.unique(offset_fronts) if i != 0]

    best_id_so_far = -1
    closest_offset_sample_idx = sys.maxsize
    for offset_front_id in offset_front_ids:
        # get all that offset front's indexes
        offset_front_idxs = _get_front_idxs_from_id(offset_fronts, offset_front_id)

        # get the sample indexes
        offset_front_sample_idxs = [s for _f, s in offset_front_idxs]

        # if each sample index is greater than onset_sample_idx, keep this offset front if it is the best one so far
        min_sample_idx = min(offset_front_sample_idxs)
        if min_sample_idx > onset_sample_idx and min_sample_idx < closest_offset_sample_idx:
            closest_offset_sample_idx = min_sample_idx
            best_id_so_far = offset_front_id

    assert best_id_so_far > 1 or best_id_so_far == -1
    return best_id_so_far

def _get_offset_front_id_after_onset_front(onset_front_id, onset_fronts, offset_fronts):
    """
    Get the ID corresponding to the offset which occurs first after the given onset_front_id.
    By `first` I mean the front which contains the offset which is closest to the latest point
    in the onset front. By `after`, I mean that the offset must contain only offsets which
    occur after the latest onset in the onset front.

    If there is no appropriate offset front, the id returned is -1.
    """
    # get the onset idxs for this front
    onset_idxs = _get_front_idxs_from_id(onset_fronts, onset_front_id)

    # get the sample idxs for this front
    onset_sample_idxs = [s for _f, s in onset_idxs]

    # get the latest onset in this onset front
    latest_onset_in_front = max(onset_sample_idxs)

    offset_front_id_after_this_onset_front = _get_offset_front_id_after_onset_sample_idx(latest_onset_in_front, offset_fronts)

    return int(offset_front_id_after_this_onset_front)

def _match_offset_front_id_to_onset_front_id(onset_front_id, onset_fronts, offset_fronts, onsets, offsets):
    """
    Find all offset fronts which are composed of at least one offset which corresponds to one of the onsets in the
    given onset front.
    The offset front which contains the most of such offsets is the match.
    If there are no such offset fronts, return -1.
    """
    # find all offset fronts which are composed of at least one offset which corresponds to one of the onsets in the onset front
    # the offset front which contains the most of such offsets is the match

    # get the onsets that make up front_id
    onset_idxs = _get_front_idxs_from_id(onset_fronts, onset_front_id)

    # get the offsets that match the onsets in front_id
    offset_idxs = [_lookup_offset_by_onset_idx(i, onsets, offsets) for i in onset_idxs]

    # get all offset_fronts which contain at least one of these offsets
    candidate_offset_front_ids = set([int(offset_fronts[f, i]) for f, i in offset_idxs])

    # It is possible that offset_idxs contains offset indexes that correspond to offsets that did not
    # get formed into a front - those will have a front ID of 0. Remove them.
    candidate_offset_front_ids = [id for id in candidate_offset_front_ids if id != 0]

    if candidate_offset_front_ids:
        chosen_offset_front_id = _choose_front_id_from_candidates(candidate_offset_front_ids, offset_fronts, offset_idxs)
    else:
        chosen_offset_front_id = _get_offset_front_id_after_onset_front(onset_front_id, onset_fronts, offset_fronts)

    return chosen_offset_front_id

def _match_fronts(onset_fronts, offset_fronts, onsets, offsets):
    """
    Returns a segmentation mask, which looks like this:
    frequency 1: 0 0 4 4 4 4 4 0 0 5 5 5
    frequency 2: 0 4 4 4 4 4 0 0 0 0 5 5
    frequency 3: 0 4 4 4 4 4 4 4 5 5 5 5

    That is, each item in the array is either a 0 (not part of a segment) or a positive
    integer which indicates which segment the sample in that frequency band belongs to.
    """
    onset_front_ids = [int(i) for i in np.unique(onset_fronts) if i != 0]  # 0 means a non-front sample
    for onset_front_id in onset_front_ids:
        # TODO: This algorithm needs to be reworked so that we do it frequency by frequency, so that
        #       we simply claim all samples in a frequency from onset to its nearest offset front,
        #       then follow that offset front down until it ends or until the onset front ends.
        #       If the offset front extends past the onset front's end, we zero the rest of the offset front.
        #       If the onset front extends past the offset front's end, we run the algorithm again, starting
        #       at the first onset front frequency that extends past the offset front.
        chosen_offset_front_id = _match_offset_front_id_to_onset_front_id(onset_front_id, onset_fronts, offset_fronts, onsets, offsets)
        # TODO
        # All items from onset_front_idx to chosen_offset_front_idx are updated to be onset_front_id
        # E.g.
        # [ N 0 0 0 0 0 0 0 0 F ]
        # [ 0 N 0 0 0 0 0 0 F 0 ]
        # [ 0 N 0 0 0 0 0 0 0 F ]
        # [ 0 0 0 0 0 0 0 0 0 F ]
        # Becomes
        # [ N N N N N N N N N N ]
        # [ 0 N N N N N N N N 0 ]
        # [ 0 N N N N N N N N N ]
        # [ 0 0 0 0 0 0 0 0 0 0 ] <-- Note that edges that extend beyond the overlap are removed

        # After doing this, if our onset front extends beyond the chosen offset front,
        # find another offset front to match, this time in the frequencies remaining.
        # I.e.:
        # [ N 0 0 0 0 0 0 0 0 F ]
        # [ 0 N 0 0 0 0 0 0 F 0 ]
        # [ 0 N 0 0 0 0 0 0 0 F ]
        # [ N 0 0 0 0 0 0 0 0 0 ]
        # [ N 0 0 0 F 0 0 0 0 0 ]
        # [ 0 N 0 0 F 0 0 0 0 0 ]
        # Becomes
        # [ N N N N N N N N N N ]
        # [ 0 N N N N N N N N 0 ]
        # [ 0 N N N N N N N N N ]
        # [ N 0 0 0 0 0 0 0 0 0 ]  <-- Note that the onset front extends past the segment, so do it again, starting here
        # [ N 0 0 0 F 0 0 0 0 0 ]
        # [ 0 N 0 0 F 0 0 0 0 0 ]
 
 
        # Keep doing this until you have matched the entire onset front.
 
    # return segmentation_mask
