"""
This module extracts out a bunch of the Auditory Scene Analysis (ASA)
logic, which has grown to be a little unwieldy in the AudioSegment class.
"""
import multiprocessing
import numpy as np
import scipy.signal as signal
import sys

def _plot(frequencies, spect, title, fn=None):
    import matplotlib.pyplot as plt
    for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
        plt.subplot(spect.shape[0], 1, index + 1)
        if index == 0:
            plt.title(title)
        plt.ylabel("{0:.0f}".format(freq))
        plt.plot(row)
        if fn is not None:
            fn(plt, freq, index, row)

def visualize_time_domain(seg, title=""):
    import matplotlib.pyplot as plt
    plt.plot(seg)
    plt.title(title)
    plt.show()
    plt.clf()

def visualize(spect, frequencies, title=""):
    import matplotlib.pyplot as plt
    _plot(frequencies, spect, title)
    plt.show()
    plt.clf()

def visualize_peaks_and_valleys(peaks, valleys, spect, frequencies):
    import matplotlib.pyplot as plt
    # Reverse everything to make it have the low fs at the bottom of the figure
    peaks = peaks[::-1, :]
    valleys = valleys[::-1, :]
    def _plot_pandv(p, _freq, index, row):
        these_peaks = peaks[index]
        peak_values = these_peaks * row  # Mask off anything that isn't a peak
        these_valleys = valleys[index]
        valley_values = these_valleys * row  # Mask off anything that isn't a valley
        p.plot(peak_values, 'ro')
        p.plot(valley_values, 'bo')
    _plot(frequencies, spect, "Peaks (red) and Valleys (blue)", _plot_pandv)
    plt.show()
    plt.clf()

def visualize_fronts(onsets, offsets, spect, frequencies):
    import matplotlib.pyplot as plt
    # Reverse everything to make it have the low fs at the bottom of the figure
    onsets = onsets[::-1, :]
    offsets = offsets[::-1, :]
    def _plot_fronts(p, _freq, index, row):
        # Cycle through all the different onsets and offsets and plot them each
        colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
        nonzero_indexes_onsets = np.reshape(np.where(onsets[index, :] != 0), (-1,))
        for x in nonzero_indexes_onsets:
            id = int(onsets[index][x])
            p.axvline(x=x, color=colors[id % len(colors)], linestyle='--')
        nonzero_indexes_offsets = np.reshape(np.where(offsets[index, :] != 0), (-1,))
        for x in nonzero_indexes_offsets:
            id = int(offsets[index][x])
            p.axvline(x=x, color=colors[id % len(colors)], linestyle='-')
    _plot(frequencies, spect, "Fronts", _plot_fronts)
    plt.show()
    plt.clf()

def visualize_segmentation_mask(segmentation, spect, frequencies, mode='new'):
    import matplotlib.pyplot as plt
    if mode.lower() == 'old':
        # Reverse segmentation mask's frequency dimension so that low fs is at the bottome
        segmentation = np.copy(segmentation)
        segmentation = segmentation[::-1, :]
        def _plot_seg(p, freq, index, row):
            colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
            segment_ids_in_this_frequency = np.reshape(np.where(segmentation[index, :] != 0), (-1,))
            for x in segment_ids_in_this_frequency:
                id = int(segmentation[index][x])
                plot_a_line = False
                if x == 0:  # if this is the first item in the frequency, then it must be a start of the mask
                    plot_a_line = True
                elif int(segmentation[index][x - 1]) != id:  # this is the start of a mask in this frequency
                    plot_a_line = True
                elif x == len(row) - 1:  # this is the very last sample, it must be the end of a mask
                    plot_a_line = True
                elif int(segmentation[index][x + 1]) != id:  # this is the last sample of the mask in this frequency
                    plot_a_line = True

                if plot_a_line:
                    p.axvline(x=x, color=colors[id % len(colors)], linestyle="--")
        _plot(frequencies, spect, "Segmentation Mask", _plot_seg)
    elif mode.lower() == 'new':
        times = np.arange(spect.shape[1])
        spect = spect * 10000
        plt.pcolormesh(times, frequencies, segmentation)
        plt.show()
    else:
        raise ValueError("Mode must be one of ('new', 'old')")
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
                if not np.any(offset_choices):
                    continue
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
    if id == -1:
        # This is the only special case.
        # -1 is the index of the catch-all final column offset front.
        freq_idxs = np.arange(fronts.shape[0], dtype=np.int64)
        sample_idxs = np.ones(len(freq_idxs), dtype=np.int64) * (fronts.shape[1] - 1)
    else:
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

def _get_consecutive_portions_of_front(front):
    """
    Yields lists of the form [(f, s), (f, s)], one at a time from the given front (which is a list of the same form),
    such that each list yielded is consecutive in frequency.
    """
    last_f = None
    ls = []
    for f, s in front:
        if last_f is not None and f != last_f + 1:
            yield ls
            ls = []
        ls.append((f, s))
        last_f = f
    yield ls

def _get_consecutive_and_overlapping_fronts(onset_fronts, offset_fronts, onset_front_id, offset_front_id):
    """
    Gets an onset_front and an offset_front such that they both occupy at least some of the same
    frequency channels, then returns the portion of each that overlaps with the other.
    """
    # Get the onset front of interest
    onset_front = _get_front_idxs_from_id(onset_fronts, onset_front_id)

    # Get the offset front of interest
    offset_front = _get_front_idxs_from_id(offset_fronts, offset_front_id)

    # Keep trying consecutive portions of this onset front until we find a consecutive portion
    # that overlaps with part of the offset front
    consecutive_portions_of_onset_front = [c for c in _get_consecutive_portions_of_front(onset_front)]
    for consecutive_portion_of_onset_front in consecutive_portions_of_onset_front:
        # Only get the segment of this front that overlaps in frequencies with the onset front of interest
        onset_front_frequency_indexes = [f for f, _ in consecutive_portion_of_onset_front]
        overlapping_offset_front = [(f, s) for f, s in offset_front if f in onset_front_frequency_indexes]

        # Only get as much of this overlapping portion as is actually consecutive
        for consecutive_portion_of_offset_front in _get_consecutive_portions_of_front(overlapping_offset_front):
            if consecutive_portion_of_offset_front:
                # Just return the first one we get - if we get any it means we found a portion of overlap
                return consecutive_portion_of_onset_front, consecutive_portion_of_offset_front
    return [], []  # These two fronts have no overlap


def _update_segmentation_mask(segmentation_mask, onset_fronts, offset_fronts, onset_front_id, offset_front_id_most_overlap):
    """
    Returns an updated segmentation mask such that the input `segmentation_mask` has been updated by segmenting between
    `onset_front_id` and `offset_front_id`, as found in `onset_fronts` and `offset_fronts`, respectively.

    This function also returns the onset_fronts and offset_fronts matrices, updated so that any fronts that are of
    less than 3 channels wide are removed.

    This function also returns a boolean value indicating whether the onset channel went to completion.

    Specifically, segments by doing the following:

    - Going across frequencies in the onset_front,
    - add the segment mask ID (the onset front ID) to all samples between the onset_front and the offset_front,
      if the offset_front is in that frequency.

    Possible scenarios:

    Fronts line up completely:

    ::

        |   |       S S S
        |   |  =>   S S S
        |   |       S S S
        |   |       S S S

    Onset front starts before offset front:

    ::

        |           |
        |   |       S S S
        |   |  =>   S S S
        |   |       S S S

    Onset front ends after offset front:

    ::

        |   |       S S S
        |   |  =>   S S S
        |   |       S S S
        |           |

    Onset front starts before and ends after offset front:

    ::

        |           |
        |   |  =>   S S S
        |   |       S S S
        |           |

    The above three options in reverse:

    ::

            |       |S S|           |
        |S S|       |S S|       |S S|
        |S S|       |S S|       |S S|
        |S S|           |           |

    There is one last scenario:

    ::

        |   |
        \   /
         \ /
         / \
        |   |

    Where the offset and onset fronts cross one another. If this happens, we simply
    reverse the indices and accept:

    ::

        |sss|
        \sss/
         \s/
         /s\
        |sss|

    The other option would be to destroy the offset front from the crossover point on, and
    then search for a new offset front for the rest of the onset front.
    """
    # Get the portions of the onset and offset fronts that overlap and are consecutive
    onset_front_overlap, offset_front_overlap = _get_consecutive_and_overlapping_fronts(onset_fronts, offset_fronts, onset_front_id, offset_front_id_most_overlap)
    onset_front = _get_front_idxs_from_id(onset_fronts, onset_front_id)
    offset_front = _get_front_idxs_from_id(offset_fronts, offset_front_id_most_overlap)
    msg = "Onset front {} and offset front {} result in consecutive overlapping portions of (on) {} and (off) {}, one of which is empty".format(
        onset_front, offset_front, onset_front_overlap, offset_front_overlap
    )
    assert onset_front_overlap, msg
    assert offset_front_overlap, msg
    onset_front = onset_front_overlap
    offset_front = offset_front_overlap

    # Figure out which frequencies will go in the segment
    flow_on, _slow_on = onset_front[0]
    fhigh_on, _shigh_on = onset_front[-1]
    flow_off, _slow_off = offset_front[0]
    fhigh_off, _shigh_off = offset_front[-1]
    flow = max(flow_on, flow_off)
    fhigh = min(fhigh_on, fhigh_off)

    # Update all the masks with the segment
    for fidx, _freqchan in enumerate(segmentation_mask[flow:fhigh + 1, :], start=flow):
        assert fidx >= flow, "Frequency index is {}, but we should have started at {}".format(fidx, flow)
        assert (fidx - flow) < len(onset_front), "Frequency index {} minus starting frequency {} is too large for nfrequencies {} in onset front {}".format(
            fidx, flow, len(onset_front), onset_front
        )
        assert (fidx - flow) < len(offset_front), "Frequency index {} minus starting frequency {} is too large for nfrequencies {} in offset front {}".format(
            fidx, flow, len(offset_front), offset_front
        )
        _, beg = onset_front[fidx - flow]
        _, end = offset_front[fidx - flow]
        if beg > end:
            end, beg = beg, end
        assert end >= beg
        segmentation_mask[fidx, beg:end + 1] = onset_front_id
        onset_fronts[fidx, (beg + 1):(end + 1)] = 0
        offset_fronts[fidx, (beg + 1):(end + 1)] = 0
    nfreqs_used_in_onset_front = (fidx - flow) + 1

    # Update the other masks to delete fronts that have been used
    indexes = np.arange(flow, fhigh + 1, 1, dtype=np.int64)
    onset_front_sample_idxs_across_freqs = np.array([s for _, s in onset_front])
    onset_front_sample_idxs_across_freqs_up_to_break = onset_front_sample_idxs_across_freqs[:nfreqs_used_in_onset_front]
    offset_front_sample_idxs_across_freqs = np.array([s for _, s in offset_front])
    offset_front_sample_idxs_across_freqs_up_to_break = offset_front_sample_idxs_across_freqs[:nfreqs_used_in_onset_front]

    ## Remove the offset front from where we started to where we ended
    offset_fronts[indexes[:nfreqs_used_in_onset_front], offset_front_sample_idxs_across_freqs_up_to_break] = 0

    ## Remove the onset front from where we started to where we ended
    onset_fronts[indexes[:nfreqs_used_in_onset_front], onset_front_sample_idxs_across_freqs_up_to_break] = 0

    # Determine if we matched the entire onset front by checking if there is any more of this onset front in onset_fronts
    whole_onset_front_matched = onset_front_id not in np.unique(onset_fronts)

    return whole_onset_front_matched

def _front_id_from_idx(front, index):
    """
    Returns the front ID found in `front` at the given `index`.

    :param front:               An onset or offset front array of shape [nfrequencies, nsamples]
    :index:                     A tuple of the form (frequency index, sample index)
    :returns:                   The ID of the front or -1 if not found in `front` and the item at `onsets_or_offsets[index]`
                                is not a 1.
    """
    fidx, sidx = index
    id = front[fidx, sidx]
    if id == 0:
        return -1
    else:
        return id

def _get_front_ids_one_at_a_time(onset_fronts):
    """
    Yields one onset front ID at a time until they are gone. All the onset fronts from a
    frequency channel are yielded, then all of the next channel's, etc., though one at a time.
    """
    yielded_so_far = set()
    for row in onset_fronts:
        for id in row:
            if id != 0 and id not in yielded_so_far:
                yield id
                yielded_so_far.add(id)

def _get_corresponding_offsets(onset_fronts, onset_front_id, onsets, offsets):
    """
    Gets the offsets that occur as close as possible to the onsets in the given onset-front.
    """
    corresponding_offsets = []
    for index in _get_front_idxs_from_id(onset_fronts, onset_front_id):
        offset_fidx, offset_sidx = _lookup_offset_by_onset_idx(index, onsets, offsets)
        corresponding_offsets.append((offset_fidx, offset_sidx))
    return corresponding_offsets

def _get_all_offset_fronts_from_offsets(offset_fronts, corresponding_offsets):
    """
    Returns all the offset fronts that are composed of at least one of the given offset indexes.
    Also returns a dict of the form {offset_front_id: ntimes saw}
    """
    all_offset_fronts_of_interest = []
    ids_ntimes_seen = {}
    for offset_index in corresponding_offsets:
        offset_id = _front_id_from_idx(offset_fronts, offset_index)
        if offset_id not in ids_ntimes_seen:
            offset_front_idxs = _get_front_idxs_from_id(offset_fronts, offset_id)
            all_offset_fronts_of_interest.append(offset_front_idxs)
            ids_ntimes_seen[offset_id] = 1
        else:
            ids_ntimes_seen[offset_id] += 1
    return all_offset_fronts_of_interest, ids_ntimes_seen

def _remove_overlaps(segmentation_mask, fronts):
    """
    Removes all points in the fronts that overlap with the segmentation mask.
    """
    fidxs, sidxs = np.where((segmentation_mask != fronts) & (segmentation_mask != 0) & (fronts != 0))
    fronts[fidxs, sidxs] = 0

def _match_fronts(onset_fronts, offset_fronts, onsets, offsets, debug=False):
    """
    Returns a segmentation mask, which looks like this:
    frequency 1: 0 0 4 4 4 4 4 0 0 5 5 5
    frequency 2: 0 4 4 4 4 4 0 0 0 0 5 5
    frequency 3: 0 4 4 4 4 4 4 4 5 5 5 5

    That is, each item in the array is either a 0 (not part of a segment) or a positive
    integer which indicates which segment the sample in that frequency band belongs to.
    """
    def printd(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    # Make copies of everything, so we can do whatever we want with them
    onset_fronts = np.copy(onset_fronts)
    offset_fronts = np.copy(offset_fronts)
    onsets = np.copy(onsets)
    offsets = np.copy(offsets)

    # This is what we will return
    segmentation_mask = np.zeros_like(onset_fronts)

    # - Take the first frequency in the onset_fronts matrix
    #     [ s s s s s s s s s] <-- This frequency
    #     [ s s s s s s s s s]
    #     [ s s s s s s s s s]
    #     [ s s s s s s s s s]
    #     [ s s s s s s s s s]

    # - Follow it along in time like this:

    #     first sample    last sample
    #       v      -->      v
    #     [ s s s s s s s s s]
    #     [ s s s s s s s s s]
    #     [ s s s s s s s s s]
    #     [ s s s s s s s s s]
    #     [ s s s s s s s s s]

    # until you get to the first onset front in that frequency

    #     Here it is!
    #         v
    #     [ . O . . . . . . .]
    #     [ . . O . . . . . .]
    #     [ . . O . . . . . .]
    #     [ . O . . . . . . .]
    #     [ O . . . . . . . .]

    resulting_onset_fronts = np.copy(onset_fronts)
    printd("    -> Dealing with onset fronts...")
    for onset_front_id in _get_front_ids_one_at_a_time(onset_fronts):
        printd("      -> Dealing with onset front", int(onset_front_id))
        front_is_complete = False
        while not front_is_complete:
            # - Now, starting at this onset front in each frequency, find that onset's corresponding offset

            #     [ . O . . . . F . .]
            #     [ . . O . . . F . .]
            #     [ . . O . F . . . .]
            #     [ . O . F . . . . .]
            #     [ O F . . . . . . .]

            corresponding_offsets = _get_corresponding_offsets(resulting_onset_fronts, onset_front_id, onsets, offsets)

            # It is possible that onset_front_id has been removed from resulting_onset_fronts,
            # if so, skip it and move on to the next onset front (we are iterating over the original
            # to keep the iterator valid)
            if not corresponding_offsets:
                break

            # - Get all the offset fronts that are composed of at least one of these offset times

            #     [ . O . . . . 1 . .]
            #     [ . . O 3 . . 1 . .]
            #     [ . . O 3 F . 1 . .]
            #     [ . O . 3 . . . 1 .]
            #     [ O F 3 . . . . . .]

            _all_offset_fronts_of_interest, ids_ntimes_seen = _get_all_offset_fronts_from_offsets(offset_fronts, corresponding_offsets)

            # - Check how many of these offset times each of the offset fronts are composed of:

            #     [ . O . . . . Y . .]
            #     [ . . O 3 . . Y . .]
            #     [ . . O 3 F . 1 . .]
            #     [ . O . X . . . 1 .]
            #     [ O F 3 . . . . . .]

            # In this example, offset front 1 is made up of 4 offset times, 2 of which (the Y's) are offset times
            # that correspond to onsets in the onset front we are currently dealing with. Meanwhile, offset
            # front 3 is made up of 4 offset times, only one of which (the X) is one of the offsets that corresponds
            # to the onset front.

            # - Choose the offset front which matches the most offset time candidates. In this example, offset front 1
            #   is chosen because it has 2 of these offset times.
            #   If there is a tie, we choose the ID with the lower number
            ntimes_seen_sorted = sorted([(k, v) for k, v in ids_ntimes_seen.items()], key=lambda tup: (-1 * tup[1], tup[0]))
            assert len(ntimes_seen_sorted) > 0, "We somehow got an empty dict of offset front IDs"

            # Only use the special final front (the -1, catch-all front composed of final samples in each frequency) if necessary
            offset_front_id, _ntimes_seen = ntimes_seen_sorted[0]
            if offset_front_id == -1 and len(ntimes_seen_sorted) > 1:
                offset_front_id, _ntimes_seen = ntimes_seen_sorted[1]
            offset_front_id_most_overlap = offset_front_id

            # - Finally, update the segmentation mask to follow the offset
            #   front from where it first overlaps in frequency with the onset front to where it ends or to where
            #   the onset front ends, whichever happens first.

            #     [ . S S S S S S . .]
            #     [ . . S S S S S . .]
            #     [ . . S S S S S . .]
            #     [ . S S S S S S S .]
            #     [ O F 3 . . . . . .]  <-- This frequency has not yet been matched with an offset front
            front_is_complete = _update_segmentation_mask(segmentation_mask,
                                                            resulting_onset_fronts,
                                                            offset_fronts,
                                                            onset_front_id,
                                                            offset_front_id_most_overlap)

            # Remove any onsets that are covered by the new segmentation mask
            _remove_overlaps(segmentation_mask, resulting_onset_fronts)

            # Remove any offsets that are covered by the new segmentaion mask
            _remove_overlaps(segmentation_mask, offset_fronts)

            # - Repeat this algorithm, restarting in the first frequency channel that did not match (the last frequency in
            #   the above example). Do this until you have finished with this onset front.

        # - Repeat for each onset front in the rest of this frequency
        # - Repeat for each frequency

    return segmentation_mask


def _remove_fronts_that_are_too_small(fronts, size):
    """
    Removes all fronts from `fronts` which are strictly smaller than
    `size` consecutive frequencies in length.
    """
    ids = np.unique(fronts)
    for id in ids:
        if id == 0 or id == -1:
            continue
        front = _get_front_idxs_from_id(fronts, id)
        if len(front) < size:
            indexes = ([f for f, _ in front], [s for _, s in front])
            fronts[indexes] = 0

def _break_poorly_matched_fronts(fronts, threshold=0.1, threshold_overlap_samples=3):
    """
    For each onset front, for each frequency in that front, break the onset front if the signals
    between this frequency's onset and the next frequency's onset are not similar enough.

    Specifically:
    If we have the following two frequency channels, and the two O's are part of the same onset front,

    ::

        [ . O . . . . . . . . . . ]
        [ . . . . O . . . . . . . ]

    We compare the signals x and y:

    ::

        [ . x x x x . . . . . . . ]
        [ . y y y y . . . . . . . ]

    And if they are not sufficiently similar (via a DSP correlation algorithm), we break the onset
    front between these two channels.

    Once this is done, remove any onset fronts that are less than 3 channels wide.
    """
    assert threshold_overlap_samples > 0, "Number of samples of overlap must be greater than zero"
    breaks_after = {}
    for front_id in _get_front_ids_one_at_a_time(fronts):
        front = _get_front_idxs_from_id(fronts, front_id)
        for i, (f, s) in enumerate(front):
            if i < len(front) - 1:
                # Get the signal from f, s to f, s+1 and the signal from f+1, s to f+1, s+1
                next_f, next_s = front[i + 1]
                low_s = min(s, next_s)
                high_s = max(s, next_s)
                sig_this_f = fronts[f, low_s:high_s]
                sig_next_f = fronts[next_f, low_s:high_s]
                assert len(sig_next_f) == len(sig_this_f)

                if len(sig_next_f) > threshold_overlap_samples:
                    # If these two signals are not sufficiently close in form, this front should be broken up
                    correlation = signal.correlate(sig_this_f, sig_next_f, mode='same')
                    assert len(correlation) > 0
                    correlation = correlation / max(correlation + 1E-9)
                    similarity = np.sum(correlation) / len(correlation)
                    # TODO: the above stuff probably needs to be figured out
                    if similarity < threshold:
                        if front_id in breaks_after:
                            breaks_after[front_id].append((f, s))
                        else:
                            breaks_after[front_id] = []

    # Now update the fronts matrix by breaking up any fronts at the points we just identified
    # and assign the newly created fronts new IDs
    taken_ids = sorted(np.unique(fronts))
    next_id = taken_ids[-1] + 1
    for id in breaks_after.keys():
        for f, s in breaks_after[id]:
            fidxs, sidxs = np.where(fronts == id)
            idxs_greater_than_f = [fidx for fidx in fidxs if fidx > f]
            start = len(sidxs) - len(idxs_greater_than_f)
            indexes = (idxs_greater_than_f, sidxs[start:])
            fronts[indexes] = next_id
            next_id += 1

    _remove_fronts_that_are_too_small(fronts, 3)

def _update_segmentation_mask_if_overlap(toupdate, other, id, otherid):
    """
    Merges the segments specified by `id` (found in `toupdate`) and `otherid`
    (found in `other`) if they overlap at all. Updates `toupdate` accordingly.
    """
    # If there is any overlap or touching, merge the two, otherwise just return
    yourmask = other == otherid
    mymask = toupdate == id
    overlap_exists = np.any(yourmask & mymask)
    if not overlap_exists:
        return

    yourfidxs, yoursidxs = np.where(other == otherid)
    toupdate[yourfidxs, yoursidxs] = id

def _segments_are_adjacent(seg1, seg2):
    """
    Checks if seg1 and seg2 are adjacent at any point. Each is a tuple of the form
    (fidxs, sidxs).
    """
    # TODO: This is unnacceptably slow
    lsf1, lss1 = seg1
    lsf2, lss2 = seg2
    for i, f1 in enumerate(lsf1):
        for j, f2 in enumerate(lsf2):
            if f1 <= f2 + 1 and f1 >= f2 - 1:
                # Frequencies are a match, are samples?
                if lss1[i] <= lss2[j] + 1 and lss1[i] >= lss2[j] - 1:
                    return True
    return False

def _merge_adjacent_segments(mask):
    """
    Merges all segments in `mask` which are touching.
    """
    mask_ids = [id for id in np.unique(mask) if id != 0]
    for id in mask_ids:
        myfidxs, mysidxs = np.where(mask == id)
        for other in mask_ids:  # Ugh, brute force O(N^2) algorithm.. gross..
            if id == other:
                continue
            else:
                other_fidxs, other_sidxs = np.where(mask == other)
                if _segments_are_adjacent((myfidxs, mysidxs), (other_fidxs, other_sidxs)):
                    mask[other_fidxs, other_sidxs] = id  # This may lead to additional adjacencies, but we only do this once - otherwise too much clustering

def _integrate_segmentation_masks(segmasks):
    """
    `segmasks` should be in sorted order of [coarsest, ..., finest].

    Integrates the given list of segmentation masks together to form one segmentation mask
    by having each segment subsume ones that exist in the finer masks.
    """
    if len(segmasks) == 1:
        return segmasks

    assert len(segmasks) > 0, "Passed in empty list of segmentation masks"
    coarse_mask = np.copy(segmasks[0])
    mask_ids = [id for id in np.unique(coarse_mask) if id != 0]
    for id in mask_ids:
        for mask in segmasks[1:]:
            finer_ids = [i for i in np.unique(mask) if i != 0]
            for finer_id in finer_ids:
                _update_segmentation_mask_if_overlap(coarse_mask, mask, id, finer_id)

    # Lastly, merge all adjacent blocks, but just kidding, since this algorithm is waaaay to slow
    #_merge_adjacent_segments(coarse_mask)
    return coarse_mask

def _separate_masks_task(id, threshold, mask):
    idxs = np.where(mask == id)
    if len(idxs[0]) > threshold:
        m = np.zeros_like(mask)
        m[idxs] = id
        return m
    else:
        return None

def _separate_masks(mask, threshold=0.025):
    """
    Returns a list of segmentation masks each of the same dimension as the input one,
    but where they each have exactly one segment in them and all other samples in them
    are zeroed.

    Only bothers to return segments that are larger in total area than `threshold * mask.size`.
    """
    try:
        ncpus = multiprocessing.cpu_count()
    except NotImplementedError:
        ncpus = 2

    with multiprocessing.Pool(processes=ncpus) as pool:
        mask_ids = [id for id in np.unique(mask) if id != 0]
        thresholds = [threshold * mask.size for _ in range(len(mask_ids))]
        masks = [mask for _ in range(len(mask_ids))]
        ms = pool.starmap(_separate_masks_task, zip(mask_ids, thresholds, masks))
    return [m for m in ms if m is not None]

def _get_downsampled_indexes(arr, factor):
    fractional_component = factor - int(factor)
    indexes_to_keep = []
    overflow_counter = 0.0
    for index in range(arr.shape[1]):
        # if we overflowed, skip this item
        if overflow_counter >= 1.0:
            overflow_counter = 0.0
            continue

        # if the integer component of the factor is satisfied, skip this item
        if int(factor) != 1 and index % int(factor) == 0:
            overflow_counter += fractional_component
            continue
        elif int(factor) == 1:
            overflow_counter += fractional_component

        # Otherwise, add this index
        indexes_to_keep.append(index)
    return indexes_to_keep

def _downsample_one_or_the_other(mask, mask_indexes, stft, stft_indexes):
    """
    Takes the given `mask` and `stft`, which must be matrices of shape `frequencies, times`
    and downsamples one of them into the other one's times, so that the time dimensions
    are equal. Leaves the frequency dimension untouched.
    """
    assert len(mask.shape) == 2, "Expected a two-dimensional `mask`, but got one of {} dimensions.".format(len(mask.shape))
    assert len(stft.shape) == 2, "Expected a two-dimensional `stft`, but got one of {} dimensions.".format(len(stft.shape))

    if mask.shape[1] > stft.shape[1]:
        downsample_factor = mask.shape[1] / stft.shape[1]
        indexes = _get_downsampled_indexes(mask, downsample_factor)
        mask = mask[:, indexes]
        mask_indexes = np.array(indexes)
    elif mask.shape[1] < stft.shape[1]:
        downsample_factor = stft.shape[1] / mask.shape[1]
        indexes = _get_downsampled_indexes(stft, downsample_factor)
        stft = stft[:, indexes]
        stft_indexes = np.array(indexes)

    return mask, mask_indexes, stft, stft_indexes

def _map_segmentation_mask_to_stft_domain(mask, times, frequencies, stft_times, stft_frequencies):
    """
    Maps the given `mask`, which is in domain (`frequencies`, `times`) to the new domain (`stft_frequencies`, `stft_times`)
    and returns the result.
    """
    assert mask.shape == (frequencies.shape[0], times.shape[0]), "Times is shape {} and frequencies is shape {}, but mask is shaped {}".format(
        times.shape, frequencies.shape, mask.shape
    )
    result = np.zeros((stft_frequencies.shape[0], stft_times.shape[0]))

    if len(stft_times) > len(times):
        all_j = [j for j in range(len(stft_times))]
        idxs  = [int(i) for i in np.linspace(0, len(times) - 1, num=len(stft_times))]
        all_i = [all_j[idx] for idx in idxs]
    else:
        all_i = [i for i in range(len(times))]
        idxs  = [int(i) for i in np.linspace(0, len(stft_times) - 1, num=len(times))]
        all_j = [all_i[idx] for idx in idxs]

    for i, j in zip(all_i, all_j):
        result[:, j] = np.interp(stft_frequencies, frequencies, mask[:, i])

    return result

def _asa_task(q, masks, stft, sample_width, frame_rate, nsamples_for_each_fft):
    """
    Worker for the ASA algorithm's multiprocessing step.
    """
    # Convert each mask to (1 or 0) rather than (ID or 0)
    for mask in masks:
        mask = np.where(mask > 0, 1, 0)

    # Multiply the masks against STFTs
    masks = [mask * stft for mask in masks]

    nparrs = []
    dtype_dict = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_dict[sample_width]
    for m in masks:
        _times, nparr = signal.istft(m, frame_rate, nperseg=nsamples_for_each_fft)
        nparr = nparr.astype(dtype)
        nparrs.append(nparr)

    for m in nparrs:
        q.put(m)
    q.put("DONE")
