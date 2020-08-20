"""
This module simply exposes a wrapper of a pydub.AudioSegment object.
"""
# Disable the annoying "cannot import x" pylint
# pylint: disable=E0401

import collections
import functools
import multiprocessing
import numpy as np
import pickle
import platform
import pydub
import os
import random
import string
import subprocess
import tempfile
import warnings
import webrtcvad
from algorithms import asa
from algorithms import eventdetection as detect
from algorithms import filters
from algorithms import util
import logging
logging.basicConfig()

try:
    import librosa
    librosa_imported = True
except ImportError as e:
    logging.warning("Could not import librosa: {}. Some functionality may be disabled.".format(e))
    librosa_imported = False

try:
    import scipy.signal as signal
    scipy_imported = True
except ImportError as e:
    logging.warning("Could not import scipy: {}. Some functionality may be disabled.".format(e))
    scipy_imported = False

MS_PER_S = 1000
S_PER_MIN = 60
MS_PER_MIN = MS_PER_S * S_PER_MIN
PASCAL_TO_PCM_FUDGE = 1000
P_REF_PASCAL = 2E-5
P_REF_PCM = P_REF_PASCAL * PASCAL_TO_PCM_FUDGE

def deprecated(func):
    """
    Deprecator decorator.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return new_func

class AudioSegment:
    """
    This class is a wrapper for a pydub.AudioSegment that provides additional methods.
    """
    def __init__(self, pydubseg, name):
        self.seg = pydubseg
        self.name = name

    def __getattr__(self, attr):
        orig_attr = self.seg.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                if result == self.seg:
                    return self
                elif type(result) == pydub.AudioSegment:
                    return AudioSegment(result, self.name)
                else:
                    return  result
            return hooked
        else:
            return orig_attr

    def __len__(self):
        return len(self.seg)

    def __eq__(self, other):
        return self.seg == other

    def __ne__(self, other):
        return self.seg != other

    def __iter__(self):
        return (x for x in self.seg)

    def __getitem__(self, millisecond):
        return AudioSegment(self.seg[millisecond], self.name)

    def __add__(self, arg):
        if type(arg) == AudioSegment:
            self.seg._data = self.seg._data + arg.seg._data
        else:
            self.seg = self.seg + arg
        return self

    def __radd__(self, rarg):
        return self.seg.__radd__(rarg)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "%s: %s channels, %s bit, sampled @ %s kHz, %.3fs long" %\
            (self.name, str(self.channels), str(self.sample_width * 8),\
             str(self.frame_rate / 1000.0), self.duration_seconds)
        return s

    def __sub__(self, arg):
        if type(arg) == AudioSegment:
            self.seg = self.seg - arg.seg
        else:
            self.seg = self.seg - arg
        return self

    def __mul__(self, arg):
        if type(arg) == AudioSegment:
            self.seg = self.seg * arg.seg
        else:
            self.seg = self.seg * arg
        return self

    @property
    def spl(self):
        """
        Sound Pressure Level - defined as 20 * log10(p/p0),
        where p is the RMS of the sound wave in Pascals and p0 is
        20 micro Pascals.

        Since we would need to know calibration information about the
        microphone used to record the sound in order to transform
        the PCM values of this audiosegment into Pascals, we can't really
        give an accurate SPL measurement.

        However, we can give a reasonable guess that can certainly be used
        to compare two sounds taken from the same microphone set up.

        Be wary about using this to compare sounds taken under different recording
        conditions however, except as a simple approximation.

        Returns a scalar float representing the dB SPL of this audiosegment.
        """
        arr = self.to_numpy_array()
        if len(arr) == 0:
            return 0.0
        else:
            rms = self.rms
            ratio = rms / P_REF_PCM
            return 20.0 * np.log10(ratio + 1E-9)  # 1E-9 for numerical stability

    def filter_bank(self, lower_bound_hz=50, upper_bound_hz=8E3, nfilters=128, mode='mel'):
        """
        Returns a numpy array of shape (nfilters, nsamples), where each
        row of data is the result of bandpass filtering the audiosegment
        around a particular frequency. The frequencies are
        spaced from `lower_bound_hz` to `upper_bound_hz` and are returned with
        the np array. The particular spacing of the frequencies depends on `mode`,
        which can be either: 'linear', 'mel', or 'log'.

        .. note:: This method is an approximation of a gammatone filterbank
                  until I get around to writing an actual gammatone filterbank
                  function.

        .. code-block:: python

            # Example usage
            import audiosegment
            import matplotlib.pyplot as plt
            import numpy as np

            def visualize(spect, frequencies, title=""):
                # Visualize the result of calling seg.filter_bank() for any number of filters
                i = 0
                for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
                    plt.subplot(spect.shape[0], 1, index + 1)
                    if i == 0:
                        plt.title(title)
                        i += 1
                    plt.ylabel("{0:.0f}".format(freq))
                    plt.plot(row)
                plt.show()

            seg = audiosegment.from_file("some_audio.wav").resample(sample_rate_Hz=24000, sample_width=2, channels=1)
            spec, frequencies = seg.filter_bank(nfilters=5)
            visualize(spec, frequencies)

        .. image:: images/filter_bank.png

        :param lower_bound_hz:  The lower bound of the frequencies to use in the bandpass filters.
        :param upper_bound_hz:  The upper bound of the frequencies to use in the bandpass filters.
        :param nfilters:        The number of filters to apply. This will determine which frequencies
                                are used as well, as they are interpolated between
                                `lower_bound_hz` and `upper_bound_hz` based on `mode`.
        :param mode:            The way the frequencies are spaced. Options are: `linear`, in which case
                                the frequencies are linearly interpolated between `lower_bound_hz` and
                                `upper_bound_hz`, `mel`, in which case the mel frequencies are used,
                                or `log`, in which case they are log-10 spaced.
        :returns:               A numpy array of the form (nfilters, nsamples), where each row is the
                                audiosegment, bandpass-filtered around a particular frequency,
                                and the list of frequencies. I.e., returns (spec, freqs).
        """
        # Logspace to get all the frequency channels we are after
        data = self.to_numpy_array()
        if mode.lower() == 'mel' and not librosa_imported:
            raise ValueError("Cannot use 'mel' when librosa not present. Install the librosa python package and try again.")
        elif mode.lower() == 'mel':
            frequencies = librosa.core.mel_frequencies(n_mels=nfilters, fmin=lower_bound_hz, fmax=upper_bound_hz)
        elif mode.lower() == 'linear':
            frequencies = np.linspace(lower_bound_hz, upper_bound_hz, num=nfilters, endpoint=True)
        elif mode.lower() == 'log':
            start = np.log10(lower_bound_hz)
            stop = np.log10(upper_bound_hz)
            frequencies = np.logspace(start, stop, num=nfilters, endpoint=True, base=10.0)
        else:
            raise ValueError("'mode' must be one of: (mel, linear, or log), but was {}".format(mode))

        # Do a band-pass filter in each frequency
        rows = [filters.bandpass_filter(data, freq*0.8, freq*1.2, self.frame_rate) for freq in frequencies]
        rows = np.array(rows)
        spect = np.vstack(rows)
        return spect, frequencies

    def auditory_scene_analysis(self, debug=False, debugplot=False):
        """
        Algorithm based on paper: Auditory Segmentation Based on Onset and Offset Analysis,
        by Hu and Wang, 2007.

        Returns a list of AudioSegments, each of which is all the sound during this AudioSegment's duration from
        a particular source. That is, if there are several overlapping sounds in this AudioSegment, this
        method will return one AudioSegment object for each of those sounds. At least, that's the idea.

        Current version is very much in alpha, and while it shows promise, will require quite a bit more
        tuning before it can really claim to work.

        :param debug:       If `True` will print out debug outputs along the way. Useful if you want to see why it is
                            taking so long.
        :param debugplot:   If `True` will use Matplotlib to plot the resulting spectrogram masks in Mel frequency scale.
        :returns:           List of AudioSegment objects, each of which is from a particular sound source.
        """
        if not scipy_imported:
            raise NotImplementedError("This method requires the STFT, which requires scipy. But scipy could not be found on this system.")

        def printd(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # Create a spectrogram from a filterbank: [nfreqs, nsamples]
        printd("Making filter bank. This takes a little bit.")
        spect, frequencies = self.filter_bank(nfilters=128)  # TODO: replace with correct number from paper

        # Half-wave rectify each frequency channel so that each value is 0 or greater - we are looking to get a temporal
        # envelope in each frequency channel
        printd("Half-wave rectifying")
        with warnings.catch_warnings():  # Ignore the annoying Numpy runtime warning for less than
            warnings.simplefilter("ignore")
            spect[spect < 0] = 0

        # Low-pass filter each frequency channel to remove a bunch of noise - we are only looking for large changes
        printd("Low pass filtering")
        low_boundary = 30
        order = 6
        spect = np.apply_along_axis(filters.lowpass_filter, 1, spect, low_boundary, self.frame_rate, order)

        # Downsample each frequency
        printd("Downsampling")
        downsample_freq_hz = 400
        if self.frame_rate > downsample_freq_hz:
            step = int(round(self.frame_rate / downsample_freq_hz))
            spect = spect[:, ::step]

        # Smoothing - we will smooth across time and frequency to further remove noise.
        # But we need to do it with several different combinations of kernels to get the best idea of what's going on
        # Scales are (sc, st), meaning (frequency scale, time scale)
        scales = [(6, 1/4), (6, 1/14), (1/2, 1/14)]

        # For each (sc, st) scale, smooth across time using st, then across frequency using sc
        gaussian = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
        gaussian_kernel = lambda sig: gaussian(np.linspace(-10, 10, len(frequencies) / 2), 0, sig)
        spectrograms = []
        printd("For each scale...")
        for sc, st in scales:
            printd("  -> Scale:", sc, st)
            printd("    -> Time and frequency smoothing")
            time_smoothed = np.apply_along_axis(filters.lowpass_filter, 1, spect, 1/st, downsample_freq_hz, 6)
            freq_smoothed = np.apply_along_axis(np.convolve, 0, time_smoothed, gaussian_kernel(sc), 'same')

            # Remove especially egregious artifacts
            printd("    -> Removing egregious filtering artifacts")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freq_smoothed[freq_smoothed > 1E3] = 1E3
                freq_smoothed[freq_smoothed < -1E3] = -1E3
            spectrograms.append(freq_smoothed)

        # Onset/Offset Detection and Matching
        segmasks = []
        printd("For each scale...")
        for spect, (sc, st) in zip(spectrograms, scales):
            printd("  -> Scale:", sc, st)
            printd("    -> Getting the onsets")
            # Compute sudden upward changes in spect, these are onsets of events
            onsets, gradients = asa._compute_peaks_or_valleys_of_first_derivative(spect)

            # Compute sudden downward changes in spect, these are offsets of events
            printd("    -> Getting the offsets")
            offsets, _ = asa._compute_peaks_or_valleys_of_first_derivative(spect, do_peaks=False)

            # Correlate offsets with onsets so that we have a 1:1 relationship
            printd("    -> Lining up the onsets and offsets")
            offsets = asa._correlate_onsets_and_offsets(onsets, offsets, gradients)

            # Create onset/offset fronts
            # Do this by connecting onsets across frequency channels if they occur within 20ms of each other
            printd("    -> Create vertical contours (fronts)")
            onset_fronts = asa._form_onset_offset_fronts(onsets, sample_rate_hz=downsample_freq_hz, threshold_ms=20)
            offset_fronts = asa._form_onset_offset_fronts(offsets, sample_rate_hz=downsample_freq_hz, threshold_ms=20)

            # Break poorly matched onset fronts
            printd("    -> Breaking onset fronts between poorly matched frequencies")
            asa._break_poorly_matched_fronts(onset_fronts)

            printd("    -> Getting segmentation mask")
            segmentation_mask = asa._match_fronts(onset_fronts, offset_fronts, onsets, offsets, debug=debug)
            segmasks.append(segmentation_mask)
            break  # TODO: We currently don't bother using the multiscale integration, so we should only do one of the scales

        # Multiscale Integration, seems to conglomerate too well and take too long
        #finished_segmentation_mask = asa._integrate_segmentation_masks(segmasks)  # TODO: doesn't work well and takes too long.
        finished_segmentation_mask = segmasks[0]
        if debugplot:
            asa.visualize_segmentation_mask(finished_segmentation_mask, spect, frequencies)

        # Change the segmentation mask's domain to that of the STFT, so we can invert it into a wave form
        ## Get the times
        times = np.arange(2 * downsample_freq_hz * len(self) / MS_PER_S)
        printd("Times vs segmentation_mask's times:", times.shape, finished_segmentation_mask.shape[1])

        ## Determine the new times and frequencies
        nsamples_for_each_fft = 2 * finished_segmentation_mask.shape[0]
        printd("Converting self into STFT")
        stft_frequencies, stft_times, stft = signal.stft(self.to_numpy_array(), self.frame_rate, nperseg=nsamples_for_each_fft)
        printd("STFTs shape:", stft.shape)
        printd("Frequencies:", stft_frequencies.shape)
        printd("Times:", stft_times.shape)

        ## Due to rounding, the STFT frequency may be one more than we want
        if stft_frequencies.shape[0] > finished_segmentation_mask.shape[0]:
            stft_frequencies = stft_frequencies[:finished_segmentation_mask.shape[0]]
            stft = stft[:stft_frequencies.shape[0], :]

        ## Downsample one into the other's times (if needed)
        finished_segmentation_mask, times, stft, stft_times = asa._downsample_one_or_the_other(stft, stft_times, finished_segmentation_mask, times)
        printd("Adjusted STFTs shape:", stft.shape)
        printd("Adjusted STFTs frequencies:", stft_frequencies.shape)
        printd("Adjusted STFTs times:", stft_times.shape)
        printd("Segmentation mask:", finished_segmentation_mask.shape)

        ## Interpolate to map the data into the new domain
        printd("Attempting to map mask of shape", finished_segmentation_mask.shape, "into shape", (stft_frequencies.shape[0], stft_times.shape[0]))
        finished_segmentation_mask = asa._map_segmentation_mask_to_stft_domain(finished_segmentation_mask, times, frequencies, stft_times, stft_frequencies)

        # Separate the mask into a bunch of single segments
        printd("Separating masks and throwing out inconsequential ones...")
        masks = asa._separate_masks(finished_segmentation_mask)
        printd("N separate masks:", len(masks))

        # If we couldn't segment into masks after thresholding,
        # there wasn't more than a single audio stream
        # Just return us as the only audio stream
        if len(masks) == 0:
            clone = from_numpy_array(self.to_numpy_array(), self.frame_rate)
            return [clone]

        # TODO: Group masks that belong together... somehow...

        # Now multiprocess the rest, since it takes forever and is easily parallelizable
        try:
            ncpus = multiprocessing.cpu_count()
        except NotImplementedError:
            ncpus = 2

        ncpus = len(masks) if len(masks) < ncpus else ncpus

        chunks = np.array_split(masks, ncpus)
        assert len(chunks) == ncpus
        queue = multiprocessing.Queue()
        printd("Using {} processes to convert {} masks into linear STFT space and then time domain.".format(ncpus, len(masks)))
        for i in range(ncpus):
            p = multiprocessing.Process(target=asa._asa_task,
                                        args=(queue, chunks[i], stft, self.sample_width, self.frame_rate, nsamples_for_each_fft),
                                        daemon=True)
            p.start()

        results = []
        dones = []
        while len(dones) < ncpus:
            item = queue.get()
            if type(item) == str and item == "DONE":
                dones.append(item)
            else:
                wav = from_numpy_array(item, self.frame_rate)
                results.append(wav)

        return results

    def detect_voice(self, prob_detect_voice=0.5):
        """
        Returns self as a list of tuples:
        [('v', voiced segment), ('u', unvoiced segment), (etc.)]

        The overall order of the AudioSegment is preserved.

        :param prob_detect_voice: The raw probability that any random 20ms window of the audio file
                                  contains voice.
        :returns: The described list.
        """
        assert self.frame_rate in (48000, 32000, 16000, 8000), "Try resampling to one of the allowed frame rates."
        assert self.sample_width == 2, "Try resampling to 16 bit."
        assert self.channels == 1, "Try resampling to one channel."

        class model_class:
            def __init__(self, aggressiveness):
                self.v = webrtcvad.Vad(int(aggressiveness))

            def predict(self, vector):
                if self.v.is_speech(vector.raw_data, vector.frame_rate):
                    return 1
                else:
                    return 0

        model = model_class(aggressiveness=2)
        pyesno = 0.3  # Probability of the next 20 ms being unvoiced given that this 20 ms was voiced
        pnoyes = 0.2  # Probability of the next 20 ms being voiced given that this 20 ms was unvoiced
        p_realyes_outputyes = 0.4  # WebRTCVAD has a very high FP rate - just because it says yes, doesn't mean much
        p_realyes_outputno  = 0.05  # If it says no, we can be very certain that it really is a no
        p_yes_raw = prob_detect_voice
        filtered = self.detect_event(model=model,
                                     ms_per_input=20,
                                     transition_matrix=(pyesno, pnoyes),
                                     model_stats=(p_realyes_outputyes, p_realyes_outputno),
                                     event_length_s=0.25,
                                     prob_raw_yes=p_yes_raw)
        ret = []
        for tup in filtered:
            t = ('v', tup[1]) if tup[0] == 'y' else ('u', tup[1])
            ret.append(t)
        return ret

    def dice(self, seconds, zero_pad=False):
        """
        Cuts the AudioSegment into `seconds` segments (at most). So for example, if seconds=10,
        this will return a list of AudioSegments, in order, where each one is at most 10 seconds
        long. If `zero_pad` is True, the last item AudioSegment object will be zero padded to result
        in `seconds` seconds.

        :param seconds: The length of each segment in seconds. Can be either a float/int, in which case
                        `self.duration_seconds` / `seconds` are made, each of `seconds` length, or a
                        list-like can be given, in which case the given list must sum to
                        `self.duration_seconds` and each segment is specified by the list - e.g.
                        the 9th AudioSegment in the returned list will be `seconds[8]` seconds long.
        :param zero_pad: Whether to zero_pad the final segment if necessary. Ignored if `seconds` is
                         a list-like.
        :returns: A list of AudioSegments, each of which is the appropriate number of seconds long.
        :raises: ValueError if a list-like is given for `seconds` and the list's durations do not sum
                 to `self.duration_seconds`.
        """
        try:
            total_s = sum(seconds)
            if not (self.duration_seconds <= total_s + 1 and self.duration_seconds >= total_s - 1):
                raise ValueError("`seconds` does not sum to within one second of the duration of this AudioSegment.\
                                 given total seconds: %s and self.duration_seconds: %s" % (total_s, self.duration_seconds))
            starts = []
            stops = []
            time_ms = 0
            for dur in seconds:
                starts.append(time_ms)
                time_ms += dur * MS_PER_S
                stops.append(time_ms)
            zero_pad = False
        except TypeError:
            # `seconds` is not a list
            starts = range(0, int(round(self.duration_seconds * MS_PER_S)), int(round(seconds * MS_PER_S)))
            stops = (min(self.duration_seconds * MS_PER_S, start + seconds * MS_PER_S) for start in starts)
        outs = [self[start:stop] for start, stop in zip(starts, stops)]
        out_lens = [out.duration_seconds for out in outs]
        # Check if our last slice is within one ms of expected - if so, we don't need to zero pad
        if zero_pad and not (out_lens[-1] <= seconds * MS_PER_S + 1 and out_lens[-1] >= seconds * MS_PER_S - 1):
            num_zeros = self.frame_rate * (seconds * MS_PER_S - out_lens[-1])
            outs[-1] = outs[-1].zero_extend(num_samples=num_zeros)
        return outs

    def detect_event(self, model, ms_per_input, transition_matrix, model_stats, event_length_s,
                     start_as_yes=False, prob_raw_yes=0.5):
        """
        A list of tuples of the form [('n', AudioSegment), ('y', AudioSegment), etc.] is returned, where tuples
        of the form ('n', AudioSegment) are the segments of sound where the event was not detected,
        while ('y', AudioSegment) tuples were the segments of sound where the event was detected.

        .. code-block:: python

            # Example usage
            import audiosegment
            import keras
            import keras.models
            import numpy as np
            import sys

            class Model:
                def __init__(self, modelpath):
                    self.model = keras.models.load_model(modelpath)

                def predict(self, seg):
                    _bins, fft_vals = seg.fft()
                    fft_vals = np.abs(fft_vals) / len(fft_vals)
                    predicted_np_form = self.model.predict(np.array([fft_vals]), batch_size=1)
                    prediction_as_int = int(round(predicted_np_form[0][0]))
                    return prediction_as_int

            modelpath = sys.argv[1]
            wavpath = sys.argv[2]
            model = Model(modelpath)
            seg = audiosegment.from_file(wavpath).resample(sample_rate_Hz=32000, sample_width=2, channels=1)
            pyes_to_no = 0.3  # The probability of one 30 ms sample being an event, and the next one not
            pno_to_yes = 0.2  # The probability of one 30 ms sample not being an event, and the next one yes
            ptrue_pos_rate = 0.8  # The true positive rate (probability of a predicted yes being right)
            pfalse_neg_rate = 0.3  # The false negative rate (probability of a predicted no being wrong)
            raw_prob = 0.7  # The raw probability of seeing the event in any random 30 ms slice of this file
            events = seg.detect_event(model, ms_per_input=30, transition_matrix=[pyes_to_no, pno_to_yes],
                                      model_stats=[ptrue_pos_rate, pfalse_neg_rate], event_length_s=0.25,
                                      prob_raw_yes=raw_prob)
            nos = [event[1] for event in events if event[0] == 'n']
            yeses = [event[1] for event in events if event[0] == 'y']
            if len(nos) > 1:
                notdetected = nos[0].reduce(nos[1:])
                notdetected.export("notdetected.wav", format="WAV")
            if len(yeses) > 1:
                detected = yeses[0].reduce(yeses[1:])
                detected.export("detected.wav", format="WAV")


        :param model:               The model. The model must have a predict() function which takes an AudioSegment
                                    of `ms_per_input` number of ms and which outputs 1 if the audio event is detected
                                    in that input, and 0 if not. Make sure to resample the AudioSegment to the right
                                    values before calling this function on it.

        :param ms_per_input:        The number of ms of AudioSegment to be fed into the model at a time. If this does not
                                    come out even, the last AudioSegment will be zero-padded.

        :param transition_matrix:   An iterable of the form: [p(yes->no), p(no->yes)]. That is, the probability of moving
                                    from a 'yes' state to a 'no' state and the probability of vice versa.

        :param model_stats:         An iterable of the form: [p(reality=1|output=1), p(reality=1|output=0)]. That is,
                                    the probability of the ground truth really being a 1, given that the model output a 1,
                                    and the probability of the ground truth being a 1, given that the model output a 0.

        :param event_length_s:      The typical duration of the event you are looking for in seconds (can be a float).

        :param start_as_yes:        If True, the first `ms_per_input` will be in the 'y' category. Otherwise it will be
                                    in the 'n' category.

        :param prob_raw_yes:        The raw probability of finding the event in any given `ms_per_input` vector.

        :returns:                   A list of tuples of the form [('n', AudioSegment), ('y', AudioSegment), etc.],
                                    where over the course of the list, the AudioSegment in tuple 3 picks up
                                    where the one in tuple 2 left off.

        :raises:                    ValueError if `ms_per_input` is negative or larger than the number of ms in this
                                    AudioSegment; if `transition_matrix` or `model_stats` do not have a __len__ attribute
                                    or are not length 2; if the values in `transition_matrix` or `model_stats` are not
                                    in the closed interval [0.0, 1.0].
        """
        if ms_per_input < 0 or ms_per_input / MS_PER_S > self.duration_seconds:
            raise ValueError("ms_per_input cannot be negative and cannot be longer than the duration of the AudioSegment."\
                             " The given value was " + str(ms_per_input))
        elif not hasattr(transition_matrix, "__len__") or len(transition_matrix) != 2:
            raise ValueError("transition_matrix must be an iterable of length 2.")
        elif not hasattr(model_stats, "__len__") or len(model_stats) != 2:
            raise ValueError("model_stats must be an iterable of length 2.")
        elif any([True for prob in transition_matrix if prob > 1.0 or prob < 0.0]):
            raise ValueError("Values in transition_matrix are probabilities, and so must be in the range [0.0, 1.0].")
        elif any([True for prob in model_stats if prob > 1.0 or prob < 0.0]):
            raise ValueError("Values in model_stats are probabilities, and so must be in the range [0.0, 1.0].")
        elif prob_raw_yes > 1.0 or prob_raw_yes < 0.0:
            raise ValueError("`prob_raw_yes` is a probability, and so must be in the range [0.0, 1.0]")

        # Get the yeses or nos for when the filter is triggered (when the event is on/off)
        filter_indices = [yes_or_no for yes_or_no in detect._get_filter_indices(self,
                                                                                start_as_yes,
                                                                                prob_raw_yes,
                                                                                ms_per_input,
                                                                                model,
                                                                                transition_matrix,
                                                                                model_stats)]

        # Run a homogeneity filter over the values to make local regions more self-similar (reduce noise)
        ret = detect._homogeneity_filter(filter_indices, window_size=int(round(0.25 * MS_PER_S / ms_per_input)))

        # Group the consecutive ones together
        ret = detect._group_filter_values(self, ret, ms_per_input)

        # Take the groups and turn them into AudioSegment objects
        real_ret = []
        for i, (this_yesno, next_timestamp) in enumerate(ret):
            if i > 0:
                _next_yesno, timestamp = ret[i - 1]
            else:
                timestamp = 0

            ms_per_s = 1000
            data = self[timestamp * ms_per_s:next_timestamp * ms_per_s].raw_data
            seg = AudioSegment(pydub.AudioSegment(data=data, sample_width=self.sample_width,
                                                    frame_rate=self.frame_rate, channels=self.channels), self.name)
            real_ret.append((this_yesno, seg))
        return real_ret

    def _execute_sox_cmd(self, cmd, console_output=False):
        """
        Executes a Sox command in a platform-independent manner.

        `cmd` must be a format string that includes {inputfile} and {outputfile}.
        """
        on_windows = platform.system().lower() == "windows"

        # On Windows, a temporary file cannot be shared outside the process that creates it
        # so we need to create a "permanent" file that we will use and delete afterwards
        def _get_random_tmp_file():
            if on_windows:
                rand_string = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
                tmp = self.name + "_" + rand_string
                WinTempFile = collections.namedtuple("WinTempFile", "name")
                tmp = WinTempFile(tmp)
            else:
                tmp = tempfile.NamedTemporaryFile()
            return tmp

        # Get a temp file to put our data and a temp file to store the result
        tmp = _get_random_tmp_file()
        othertmp = _get_random_tmp_file()

        # Store our data in the temp file
        self.export(tmp.name, format="WAV")

        # Write the command to sox
        stdout = stderr = subprocess.PIPE if console_output else subprocess.DEVNULL
        command = cmd.format(inputfile=tmp.name, outputfile=othertmp.name)
        res = subprocess.call(command.split(' '), stdout=stdout, stderr=stderr)
        assert res == 0, "Sox did not work as intended, or perhaps you don't have Sox installed?"

        # Create a new AudioSegment from the other temp file (where Sox put the result)
        other = AudioSegment(pydub.AudioSegment.from_wav(othertmp.name), self.name)

        # Clean up the temp files
        if on_windows:
            os.remove(tmp.name)
            os.remove(othertmp.name)
        else:
            tmp.close()
            othertmp.close()

        return other

    def filter_silence(self, duration_s=1, threshold_percentage=1, console_output=False):
        """
        Returns a copy of this AudioSegment, but whose silence has been removed.

        .. note:: This method requires that you have the program 'sox' installed.

        .. warning:: This method uses the program 'sox' to perform the task. While this is very fast for a single
                     function call, the IO may add up for large numbers of AudioSegment objects.

        :param duration_s: The number of seconds of "silence" that must be present in a row to
                           be stripped.
        :param threshold_percentage: Silence is defined as any samples whose absolute value is below
                                     `threshold_percentage * max(abs(samples in this segment))`.
        :param console_output: If True, will pipe all sox output to the console.
        :returns: A copy of this AudioSegment, but whose silence has been removed. Note that if the arguments
                  to this method result in it removing all samples from the audio, we issue a warning and return
                  a copy of the original, unchanged audio.
        """
        command = "sox {inputfile} -t wav {outputfile} silence -l 1 0.1 "\
            + str(threshold_percentage) + "% -1 " + str(float(duration_s)) + " " + str(threshold_percentage) + "%"
        try:
            result = self._execute_sox_cmd(command)
        except pydub.exceptions.CouldntDecodeError:
            warnings.warn("After silence filtering, the resultant WAV file is corrupted, and so its data cannot be retrieved. Perhaps try a smaller threshold value.", stacklevel=2)
            # Return a copy of us
            result = AudioSegment(self.seg, self.name)

        # Also check if the audio no longer has any data in it. If so, it's not terribly useful. Warn and return a copy of us.
        if len(result.to_numpy_array()) == 0:
            warnings.warn("After silence filtering, the resultant WAV file has no samples in it. Perhaps try a smaller threshold value.", stacklevel=2)
            result = AudioSegment(self.seg, self.name)

        return result

    def fft(self, start_s=None, duration_s=None, start_sample=None, num_samples=None, zero_pad=False):
        """
        Transforms the indicated slice of the AudioSegment into the frequency domain and returns the bins
        and the values.

        If neither `start_s` or `start_sample` is specified, the first sample of the slice will be the first sample
        of the AudioSegment.

        If neither `duration_s` or `num_samples` is specified, the slice will be from the specified start
        to the end of the segment.

        .. code-block:: python

            # Example for plotting the FFT using this function
            import matplotlib.pyplot as plt
            import numpy as np

            seg = audiosegment.from_file("furelise.wav")
            # Just take the first 3 seconds
            hist_bins, hist_vals = seg[1:3000].fft()
            hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
            plt.plot(hist_bins / 1000, hist_vals_real_normed)
            plt.xlabel("kHz")
            plt.ylabel("dB")
            plt.show()

        .. image:: images/fft.png

        :param start_s: The start time in seconds. If this is specified, you cannot specify `start_sample`.
        :param duration_s: The duration of the slice in seconds. If this is specified, you cannot specify `num_samples`.
        :param start_sample: The zero-based index of the first sample to include in the slice.
                             If this is specified, you cannot specify `start_s`.
        :param num_samples: The number of samples to include in the slice. If this is specified, you cannot
                            specify `duration_s`.
        :param zero_pad: If True and the combination of start and duration result in running off the end of
                         the AudioSegment, the end is zero padded to prevent this.
        :returns: np.ndarray of frequencies in Hz, np.ndarray of amount of each frequency
        :raises: ValueError If `start_s` and `start_sample` are both specified and/or if both `duration_s` and
                            `num_samples` are specified.
        """
        if start_s is not None and start_sample is not None:
            raise ValueError("Only one of start_s and start_sample can be specified.")
        if duration_s is not None and num_samples is not None:
            raise ValueError("Only one of duration_s and num_samples can be specified.")
        if start_s is None and start_sample is None:
            start_sample = 0
        if duration_s is None and num_samples is None:
            num_samples = len(self.get_array_of_samples()) - int(start_sample)

        if duration_s is not None:
            num_samples = int(round(duration_s * self.frame_rate))
        if start_s is not None:
            start_sample = int(round(start_s * self.frame_rate))

        end_sample = start_sample + num_samples  # end_sample is excluded
        if end_sample > len(self.get_array_of_samples()) and not zero_pad:
            raise ValueError("The combination of start and duration will run off the end of the AudioSegment object.")
        elif end_sample > len(self.get_array_of_samples()) and zero_pad:
            arr = np.array(self.get_array_of_samples())
            zeros = np.zeros(end_sample - len(arr))
            arr = np.append(arr, zeros)
        else:
            arr = np.array(self.get_array_of_samples())

        audioslice = np.array(arr[start_sample:end_sample])
        fft_result = np.fft.fft(audioslice)[range(int(round(num_samples/2)) + 1)]
        step_size = self.frame_rate / num_samples
        bins = np.arange(0, int(round(num_samples/2)) + 1, 1.0) * step_size
        return bins, fft_result

    def generate_frames(self, frame_duration_ms, zero_pad=True):
        """
        Yields self's data in chunks of frame_duration_ms.

        This function adapted from pywebrtc's example [https://github.com/wiseman/py-webrtcvad/blob/master/example.py].

        :param frame_duration_ms: The length of each frame in ms.
        :param zero_pad: Whether or not to zero pad the end of the AudioSegment object to get all
                         the audio data out as frames. If not, there may be a part at the end
                         of the Segment that is cut off (the part will be <= `frame_duration_ms` in length).
        :returns: A Frame object with properties 'bytes (the data)', 'timestamp (start time)', and 'duration'.
        """
        Frame = collections.namedtuple("Frame", "bytes timestamp duration")

        # (samples/sec) * (seconds in a frame) * (bytes/sample) * nchannels
        bytes_per_frame = int(self.frame_rate * (frame_duration_ms / 1000) * self.sample_width * self.channels)

        # Now round up bytes_per_frame to least common multiple of it and sample_width
        bytes_per_frame = util.lcm(bytes_per_frame, self.sample_width)

        # where we are so far in self's data (in bytes)
        offset = 0

        # where we are so far in self (in seconds)
        timestamp = 0.0

        # (bytes/frame) * (sample/bytes) * (sec/samples)
        frame_duration_s = (bytes_per_frame / self.frame_rate) / self.sample_width

        while offset + bytes_per_frame < len(self.raw_data):
            yield Frame(self.raw_data[offset:offset + bytes_per_frame], timestamp, frame_duration_s)

            timestamp += frame_duration_s
            offset += bytes_per_frame

        rest = self.raw_data[offset:]

        if zero_pad:
            zeros = bytes(bytes_per_frame - len(rest))
            yield Frame(rest + zeros, timestamp, frame_duration_s)
        elif len(rest) > 0:
            ms = (len(rest) / self.frame_rate) / self.sample_width
            yield Frame(rest, timestamp, ms)

    def generate_frames_as_segments(self, frame_duration_ms, zero_pad=True):
        """
        Does the same thing as `generate_frames`, but yields tuples of (AudioSegment, timestamp) instead of Frames.
        """
        for frame in self.generate_frames(frame_duration_ms, zero_pad=zero_pad):
            seg = AudioSegment(pydub.AudioSegment(data=frame.bytes, sample_width=self.sample_width, frame_rate=self.frame_rate, channels=self.channels), self.name)
            yield seg, frame.timestamp

    def human_audible(self):
        """
        Returns the number of seconds of human audible audio in this wave form. Note that
        I define whether a segment of the wave form is human audible based on doing an FFT
        of it and then checking for any peaks within 20 Hz to 20 kHz, a fair rule of thumb
        for human hearing thresholds.

        Also note that I make no assumptions about the SPL of the sound. This is important because
        humans can only hear something if it is loud enough (obviously), but how loud something
        needs to be to be heard depends on its frequency. I do not model this with this method.
        If you are curious how the SPL threshold changes with frequency, take a look at the
        audiogram in Hearing Thresholds by Yost and Killion, 1997 (see https://www.etymotic.com/media/publications/erl-0096-1997.pdf).

        :returns: A floating point value representing the number of seconds (with 100 ms resolution).
        """
        threshold_peak = 0.1
        threshold_wideband = 3.0
        frame_duration_s = 0.1
        n_frames_human_audible = 0
        for subsegment, _timestamp in self.generate_frames_as_segments(frame_duration_ms=frame_duration_s * 1000, zero_pad=True):
            _hist_bins, hist_vals = subsegment.fft()
            hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
            hist_vals_real_normed = hist_vals_real_normed / (np.max(hist_vals_real_normed) + 1E-9)  # numerical stability
            # If any values between 20 Hz and 20 kHz reaches an 0.x value, let's call it human audible
            vals_greater_than_point_2 = hist_vals_real_normed[hist_vals_real_normed >= threshold_peak]
            # Also, if enough of the energy of the waveform is found in the audible range, we'll call it human audible
            if np.any(vals_greater_than_point_2[20:20000]) or np.sum(hist_vals_real_normed[20:20000]) > threshold_wideband:
                n_frames_human_audible += 1
        total_seconds_audible = frame_duration_s * n_frames_human_audible
        return total_seconds_audible

    def reduce(self, others):
        """
        Reduces others into this one by concatenating all the others onto this one and
        returning the result. Does not modify self, instead, makes a copy and returns that.

        :param others: The other AudioSegment objects to append to this one.
        :returns: The concatenated result.
        """
        ret = AudioSegment(self.seg, self.name)
        selfdata = [self.seg._data]
        otherdata = [o.seg._data for o in others]
        ret.seg._data = b''.join(selfdata + otherdata)

        return ret

    def resample(self, sample_rate_Hz=None, sample_width=None, channels=None):
        """
        Returns a new AudioSegment whose data is the same as this one, but which has been resampled to the
        specified characteristics. Any parameter left None will be unchanged.

        This is mostly just a wrapper for calling pydub.AudioSegment's `set_sample_width`, `set_channels`, and
        `set_frame_rate` methods, but allows for more channels than the 1 or 2 that Pydub allows.

        :param sample_rate_Hz: The new sample rate in Hz.
        :param sample_width: The new sample width in bytes, so sample_width=2 would correspond to 16 bit (2 byte) width.
                             Note that 3-byte audio will be converted to 4-byte instead by Pydub.
        :param channels: The new number of channels.
        :returns: The newly sampled AudioSegment.
        """
        if sample_rate_Hz is None:
            sample_rate_Hz = self.frame_rate
        else:
            sample_rate_Hz = int(round(sample_rate_Hz))

        if sample_width is None:
            sample_width = self.sample_width
        else:
            sample_width = int(round(sample_width))

        if channels is None:
            channels = self.channels
        else:
            channels = int(round(channels))

        # Check args
        if sample_rate_Hz <= 0:
            raise ValueError("Sample rate must be > 0, but is {}".format(sample_rate_Hz))

        if sample_width <= 0:
            raise ValueError("Sample width must be > 0, but is {}".format(sample_width))

        if channels <= 0:
            raise ValueError("Number of channels must be > 0, but is {}".format(channels))

        if self.channels > 2:
            # Pydub does not like audio that has more than 2 channels. Handle it manually here.
            arr = self.to_numpy_array()
            dtype = arr.dtype
            if channels < self.channels:
                # Downmix by averaging (if we want half as many channels, we average every other channel together, for example)
                # This is adapted from https://stackoverflow.com/questions/30379311/fast-way-to-take-average-of-every-n-rows-in-a-npy-array
                N = int(self.channels / channels)
                arr = arr.T
                arr = np.cumsum(arr, 0)[N-1::N]/float(N)
                arr[1:] = arr[1:] - arr[:-1]
                arr = arr.astype(dtype).T
            monosegs = []
            for i in range(channels):
                targetarr = arr[:, i % arr.shape[1]]
                monoseg = from_numpy_array(targetarr, self.frame_rate).set_sample_width(sample_width).set_frame_rate(sample_rate_Hz)
                monosegs.append(monoseg)
            return from_mono_audiosegments(*monosegs)
        elif channels > 2:
            # If there are more than 2 channels, Pydub throws an exception, so handle this manually here
            seg = self.resample(sample_rate_Hz=sample_rate_Hz, sample_width=sample_width, channels=1)
            seg = from_mono_audiosegments(*[seg for _ in range(channels)])
            return seg
        else:
            return self.set_sample_width(sample_width).set_channels(channels).set_frame_rate(sample_rate_Hz)

    def __getstate__(self):
        """
        Serializes into a dict for the pickle protocol.

        :returns: The dict to pickle.
        """
        return {'name': self.name, 'seg': self.seg}

    def __setstate__(self, d):
        """
        Deserializes from a dict for the pickle protocol.

        :param d: The dict to unpickle from.
        """
        self.__dict__.update(d)

    def serialize(self):
        """
        Serializes into a bytestring.

        :returns: An object of type Bytes.
        """
        d = self.__getstate__()
        return pickle.dumps({
            'name': d['name'],
            'seg': pickle.dumps(d['seg'], protocol=-1),
        }, protocol=-1)

    def spectrogram(self, start_s=None, duration_s=None, start_sample=None, num_samples=None,
                    window_length_s=None, window_length_samples=None, overlap=0.5, window=('tukey', 0.25)):
        """
        Does a series of FFTs from `start_s` or `start_sample` for `duration_s` or `num_samples`.
        Effectively, transforms a slice of the AudioSegment into the frequency domain across different
        time bins.

        .. code-block:: python

            # Example for plotting a spectrogram using this function
            import audiosegment
            import matplotlib.pyplot as plt

            #...
            seg = audiosegment.from_file("somebodytalking.wav")
            freqs, times, amplitudes = seg.spectrogram(window_length_s=0.03, overlap=0.5)
            amplitudes = 10 * np.log10(amplitudes + 1e-9)

            # Plot
            plt.pcolormesh(times, freqs, amplitudes)
            plt.xlabel("Time in Seconds")
            plt.ylabel("Frequency in Hz")
            plt.show()

        .. image:: images/spectrogram.png

        :param start_s: The start time. Starts at the beginning if neither this nor `start_sample` is specified.
        :param duration_s: The duration of the spectrogram in seconds. Goes to the end if neither this nor
                           `num_samples` is specified.
        :param start_sample: The index of the first sample to use. Starts at the beginning if neither this nor
                             `start_s` is specified.
        :param num_samples: The number of samples in the spectrogram. Goes to the end if neither this nor
                            `duration_s` is specified.
        :param window_length_s: The length of each FFT in seconds. If the total number of samples in the spectrogram
                                is not a multiple of the window length in samples, the last window will be zero-padded.
        :param window_length_samples: The length of each FFT in number of samples. If the total number of samples in the
                                spectrogram is not a multiple of the window length in samples, the last window will
                                be zero-padded.
        :param overlap: The fraction of each window to overlap.
        :param window: See Scipy's spectrogram-function_.
                       This parameter is passed as-is directly into the Scipy spectrogram function. It's documentation is reproduced here:
                       Desired window to use. If window is a string or tuple, it is passed to get_window to generate the window values,
                       which are DFT-even by default. See get_window for a list of windows and required parameters.
                       If window is array_like it will be used directly as the window and its length must be
                       `window_length_samples`.
                       Defaults to a Tukey window with shape parameter of 0.25.
        :returns: Three np.ndarrays: The frequency values in Hz (the y-axis in a spectrogram), the time values starting
                  at start time and then increasing by `duration_s` each step (the x-axis in a spectrogram), and
                  the dB of each time/frequency bin as a 2D array of shape [len(frequency values), len(duration)].
        :raises ValueError: If `start_s` and `start_sample` are both specified, if `duration_s` and `num_samples` are both
                            specified, if the first window's duration plus start time lead to running off the end
                            of the AudioSegment, or if `window_length_s` and `window_length_samples` are either
                            both specified or if they are both not specified.

        .. _spectrogram-function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
        """
        if start_s is not None and start_sample is not None:
            raise ValueError("Only one of start_s and start_sample may be specified.")
        if duration_s is not None and num_samples is not None:
            raise ValueError("Only one of duration_s and num_samples may be specified.")
        if window_length_s is not None and window_length_samples is not None:
            raise ValueError("Only one of window_length_s and window_length_samples may be specified.")
        if window_length_s is None and window_length_samples is None:
            raise ValueError("You must specify a window length, either in window_length_s or in window_length_samples.")

        # Determine the start sample
        if start_s is None and start_sample is None:
            start_sample = 0
        elif start_s is not None:
            start_sample = int(round(start_s * self.frame_rate))

        # Determine the number of samples
        if duration_s is None and num_samples is None:
            num_samples = len(self.get_array_of_samples()) - int(start_sample)
        elif duration_s is not None:
            num_samples = int(round(duration_s * self.frame_rate))

        # Determine the number of samples per window
        if window_length_s is not None:
            window_length_samples = int(round(window_length_s * self.frame_rate))

        # Check validity of number of samples
        if start_sample + num_samples > len(self.get_array_of_samples()):
            raise ValueError("The combination of start and duration will run off the end of the AudioSegment object.")

        # Create a Numpy Array out of the correct samples
        arr = self.to_numpy_array()[start_sample:start_sample+num_samples]

        # If Scipy is present, let's use its spectrogram method, as it is much faster
        if scipy_imported:
            fs, ts, sxx = signal.spectrogram(arr, self.frame_rate, scaling='spectrum', nperseg=window_length_samples,
                                                noverlap=int(round(overlap * window_length_samples)),
                                                mode='magnitude', window=window)
            return fs, ts, sxx
        else:
            starts = []
            next_start = start_sample
            while next_start < len(self.get_array_of_samples()):
                starts.append(next_start)
                next_start = next_start + int(round(overlap * window_length_samples))

            rets = [self.fft(start_sample=start, num_samples=window_length_samples, zero_pad=True) for start in starts]
            bins = rets[0][0]
            values = [ret[1] for ret in rets]
            times = [start_sample / self.frame_rate for start_sample in starts]
            return np.array(bins), np.array(times), np.array(values)

    def to_numpy_array(self):
        """
        Returns a numpy array. The shape of this array is either (nsamples, nchannels), if nchannels
        is greater than 1, or else just (nsamples,).
        """
        dtype_dict = {
                        1: np.int8,
                        2: np.int16,
                        4: np.int32
                     }
        dtype = dtype_dict[self.sample_width]
        arr = np.array(self.get_array_of_samples(), dtype=dtype)
        return np.reshape(arr, (-1, self.channels)).squeeze()

    def zero_extend(self, duration_s=None, num_samples=None):
        """
        Adds a number of zeros (digital silence) to the AudioSegment (returning a new one).

        :param duration_s: The number of seconds of zeros to add. If this is specified, `num_samples` must be None.
        :param num_samples: The number of zeros to add. If this is specified, `duration_s` must be None.
        :returns: A new AudioSegment object that has been zero extended.
        :raises: ValueError if duration_s and num_samples are both specified.
        """
        if duration_s is not None and num_samples is not None:
            raise ValueError("`duration_s` and `num_samples` cannot both be specified.")
        elif duration_s is not None:
            num_samples = self.frame_rate * duration_s
        seg = AudioSegment(self.seg, self.name)
        zeros = silent(duration=num_samples / self.frame_rate, frame_rate=self.frame_rate)
        return zeros.overlay(seg)

def deserialize(bstr):
    """
    Attempts to deserialize a bytestring into an audiosegment.

    :param bstr: The bytestring serialized via an audiosegment's serialize() method.
    :returns: An AudioSegment object deserialized from `bstr`.
    """
    d = pickle.loads(bstr)
    seg = pickle.loads(d['seg'])
    return AudioSegment(seg, d['name'])

def empty():
    """
    Creates a zero-duration AudioSegment object.

    :returns: An empty AudioSegment object.
    """
    dubseg = pydub.AudioSegment.empty()
    return AudioSegment(dubseg, "")

def from_file(path):
    """
    Returns an AudioSegment object from the given file based on its file extension.
    If the extension is wrong, this will throw some sort of error.

    :param path: The path to the file, including the file extension.
    :returns: An AudioSegment instance from the file.
    """
    _name, ext = os.path.splitext(path)
    ext = ext.lower()[1:]
    seg = pydub.AudioSegment.from_file(path, ext)
    return AudioSegment(seg, path)

def from_mono_audiosegments(*args):
    """
    Creates a multi-channel AudioSegment out of multiple mono AudioSegments (two or more). Each mono
    AudioSegment passed in should be exactly the same number of samples.

    :returns: An AudioSegment of multiple channels formed from the given mono AudioSegments.
    """
    return AudioSegment(pydub.AudioSegment.from_mono_audiosegments(*args), "")

def from_numpy_array(nparr, framerate):
    """
    Returns an AudioSegment created from the given numpy array.

    The numpy array must have shape = (num_samples, num_channels).

    :param nparr: The numpy array to create an AudioSegment from.
    :param framerate: The sample rate (Hz) of the segment to generate.
    :returns: An AudioSegment created from the given array.
    """
    # Check args
    if nparr.dtype.itemsize not in (1, 2, 4):
        raise ValueError("Numpy Array must contain 8, 16, or 32 bit values.")

    # Determine nchannels
    if len(nparr.shape) == 1:
        nchannels = 1
    elif len(nparr.shape) == 2:
        nchannels = nparr.shape[1]
    else:
        raise ValueError("Numpy Array must be one or two dimensional. Shape must be: (num_samples, num_channels), but is {}.".format(nparr.shape))

    # Fix shape if single dimensional
    nparr = np.reshape(nparr, (-1, nchannels))

    # Create an array of mono audio segments
    monos = []
    for i in range(nchannels):
        m = nparr[:, i]
        dubseg = pydub.AudioSegment(m.tobytes(), frame_rate=framerate, sample_width=nparr.dtype.itemsize, channels=1)
        monos.append(dubseg)

    return AudioSegment(pydub.AudioSegment.from_mono_audiosegments(*monos), "")

def silent(duration=1000, frame_rate=11025):
    """
    Creates an AudioSegment object of the specified duration/frame_rate filled with digital silence.

    :param duration: The duration of the returned object in ms.
    :param frame_rate: The samples per second of the returned object.
    :returns: AudioSegment object filled with pure digital silence.
    """
    seg = pydub.AudioSegment.silent(duration=duration, frame_rate=frame_rate)
    return AudioSegment(seg, "")
