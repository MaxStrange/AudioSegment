"""
This module simply exposes a wrapper of a pydub.AudioSegment object.
"""
from __future__ import division

import collections
import numpy as np
import pydub
import os
import subprocess
import sys
import tempfile
import webrtcvad

MS_PER_S = 1000
S_PER_MIN = 60
MS_PER_MIN = MS_PER_S * S_PER_MIN

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

    def __iter__(self, other):
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

    def detect_voice(self):
        """
        Returns self as a list of tuples:
        [('v', voiced segment), ('u', unvoiced segment), (etc.)]

        The overall order of the Segment is preserved.

        :returns: The described list. Does not modify self.
        """
        assert self.frame_rate in (48000, 32000, 16000, 8000), "Try resampling to one of the allowed frame rates."
        assert self.sample_width == 2, "Try resampling to 16 bit."
        assert self.channels == 1, "Try resampling to one channel."
        def vad_collector(frame_duration_ms, padding_duration_ms, v, frames):
            """
            Collects self into segments of VAD and non VAD.

            Yields tuples, one at a time, either ('v', Segment) or ('u', Segment).
            """
            # TODO: Replace this with a Bayesian update algorithm.
            #       We should mark each frame as voiced or unvoiced, then collect them
            #       and update our belief about whether the sequence is voiced or not.
            #       Once we get to a threshold probability that we are voiced, go back
            #       and collect the last consecutive voiced frames and trigger, so that
            #       until our probability drops below a different threshold (hysteresis),
            #       we keep collecting frames as voiced regardless of what they are.
            construct_segment = lambda frames: AudioSegment(pydub.AudioSegment(data=b''.join([f.bytes for f in frames]),
                                                                          sample_width=self.sample_width,
                                                                          frame_rate=self.frame_rate,
                                                                          channels=self.channels), self.name)
            num_padding_frames = int(padding_duration_ms / frame_duration_ms)
            ring_buffer = collections.deque(maxlen=num_padding_frames)
            triggered = False
            voiced_frames = []
            for frame in frames:
                # Until we have collected 90% of num_padding_frames that we think are voiced,
                # keep shoving frames into the ring buffer. Since it is a ring buffer, it
                # drops old frames once we are at capacity.
                if not triggered:
                    ring_buffer.append(frame)
                    num_voiced = len([f for f in ring_buffer if v.is_speech(f.bytes, self.frame_rate)])
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        # We have collected enough voiced frames "in a row" to trigger
                        triggered = True
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                else:
                    # We are triggered, so collect each frame into voiced_frames until
                    # enough of them "in a row" are unvoiced.
                    voiced_frames.append(frame)
                    ring_buffer.append(frame)
                    num_unvoiced = len([f for f in ring_buffer if not v.is_speech(f.bytes, self.frame_rate)])
                    if num_unvoiced > 0.9 * ring_buffer.maxlen:
                        triggered = False
                        yield 'v', construct_segment(voiced_frames)
                        yield 'u', construct_segment(ring_buffer)
                        ring_buffer.clear()
                        voiced_frames = []
            if voiced_frames:
                yield 'v', construct_segment(voiced_frames)
            if ring_buffer:
                yield 'u', construct_segment(ring_buffer)

        aggressiveness = 2
        window_size = 20
        padding_duration_ms = 200

        frames = self.generate_frames(frame_duration_ms=window_size, zero_pad=True)
        v = webrtcvad.Vad(int(aggressiveness))
        return [tup for tup in vad_collector(window_size, padding_duration_ms, v, frames)]

    def filter_silence(self, duration_s=1, threshold_percentage=1):
        """
        Removes all silence from this segment and returns itself after modification.

        :returns: self, for convenience (self is modified in place as well)
        """
        tmp = tempfile.NamedTemporaryFile()
        othertmp = tempfile.NamedTemporaryFile()
        self.export(tmp.name, format="WAV")
        command = "sox " + tmp.name + " -t wav " + othertmp.name + " silence -l 1 0.1 "\
                   + str(threshold_percentage) + "% -1 " + str(float(duration_s)) + " " + str(threshold_percentage) + "%"
        res = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
        assert res.returncode == 0, "Sox did not work as intended, or perhaps you don't have Sox installed?"
        other = AudioSegment(pydub.AudioSegment.from_wav(othertmp.name), self.name)
        tmp.close()
        othertmp.close()
        self = other
        return self

    def fft(self, start_s=None, duration_s=None, start_sample=None, num_samples=None, zero_pad=False):
        """
        Transforms the indicated slice of the AudioSegment into the frequency domain and returns the bins
        and the values.

        If neither start_s or start_sample is specified, the first sample of the slice will be the first sample
        of the audio segment.

        If neither duration_s or num_samples is specified, the slice will be from the specified start
        to the end of the segment.

        :param start_s: The start time in seconds. If this is specified, you cannot specify start_sample.
        :param duration_s: The duration of the slice in seconds. If this is specified, you cannot specify num_samples.
        :param start_sample: The zero-based index of the first sample to include in the slice.
                             If this is specified, you cannot specify start_s.
        :param num_samples: The number of samples to include in the slice. If this is specified, you cannot
                            specify duration_s.
        :param zero_pad: If True and the combination of start and duration result in running off the end of
                         the AudioSegment, the end is zero padded to prevent this.
        :returns: np.ndarray of frequencies, np.ndarray of amount of each frequency
        :raises ValueError: If start_s and start_sample are both specified and/or if both duration_s and num_samples
                            are specified.
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
        fft_result = np.real(np.fft.fft(audioslice))[range(int(round(num_samples/2)) + 1)]
        bins = np.arange(0, int(round(num_samples/2)) + 1, 1.0) * (self.frame_rate / num_samples)
        return bins, fft_result

    def generate_frames(self, frame_duration_ms, zero_pad=True):
        """
        Yields self's data in chunks of frame_duration_ms.

        This function adapted from pywebrtc's example [https://github.com/wiseman/py-webrtcvad/blob/master/example.py].

        :param frame_duration_ms: The length of each frame in ms.
        :param zero_pad: Whether or not to zero pad the end of the Segment object to get all
                         the audio data out as frames. If not, there may be a part at the end
                         of the Segment that is cut off (the part will be <= frame_duration_ms in length).
        :returns: A Frame object with properties 'bytes (the data)', 'timestamp (start time)', and 'duration'.
        """
        Frame = collections.namedtuple("Frame", "bytes timestamp duration")

        # (samples/sec) * (seconds in a frame) * (bytes/sample)
        bytes_per_frame = int(self.frame_rate * (frame_duration_ms / 1000) * self.sample_width)
        offset = 0  # where we are so far in self's data (in bytes)
        timestamp = 0.0  # where we are so far in self (in seconds)
        # (bytes/frame) * (sample/bytes) * (sec/samples)
        frame_duration_s = (bytes_per_frame / self.frame_rate) / self.sample_width
        while offset + bytes_per_frame < len(self.raw_data):
            yield Frame(self.raw_data[offset:offset + bytes_per_frame], timestamp, frame_duration_s)
            timestamp += frame_duration_s
            offset += bytes_per_frame

        if zero_pad:
            rest = self.raw_data[offset:]
            zeros = bytes(bytes_per_frame - len(rest))
            yield Frame(rest + zeros, timestamp, frame_duration_s)

    def generate_frames_as_segments(self, frame_duration_ms, zero_pad=True):
        """
        Does the same thing as generate_frames, but yields tuples of (Segment, timestamp) instead of Frames.
        """
        for frame in self.generate_frames(frame_duration_ms, zero_pad=zero_pad):
            seg = AudioSegment(pydub.AudioSegment(data=frame.bytes, sample_width=self.sample_width, frame_rate=self.frame_rate, channels=self.channels), self.name)
            yield seg, frame.timestamp

    def reduce(self, others):
        """
        Reduces others into this one by concatenating all the others onto this one.

        :param others: The other Segment objects to append to this one.
        :returns: self, for convenience (self is modified in place as well)
        """
        self.seg._data = b''.join([self.seg._data] + [o.seg._data for o in others])

        return self

    def resample(self, sample_rate_Hz=None, sample_width=None, channels=None):
        """
        Resamples self and returns self with the new characteristics. Any paramter that is left as None will be unchanged.

        :param sample_rate_Hz: The new sample rate in Hz.
        :param sample_width: The new sample width in bytes, so sample_width=2 would correspond to 16 bit (2 byte) width.
        :param channels: The new number of channels.
        :returns: self, and also changes self in place
        """
        infile, outfile = tempfile.NamedTemporaryFile(), tempfile.NamedTemporaryFile()
        self.export(infile.name, format="wav")
        command = "sox " + infile.name + " -b" + str(sample_width * 8) + " -r " + str(sample_rate_Hz) + " -t wav " + outfile.name + " channels " + str(channels)
        res = subprocess.run(command.split(' '), stdout=subprocess.PIPE)
        res.check_returncode()
        other = AudioSegment(pydub.AudioSegment.from_wav(outfile.name), self.name)
        infile.close()
        outfile.close()
        self = other
        return self

    def trim_to_minutes(self, strip_last_seconds=False):
        """
        Does not modify self, but instead returns a list of minute-long (at most) Segment objects.

        :param strip_last_seconds: If True, this method will return minute-long segments, but the last three seconds of this Segment won't be returned.
                                   This is useful for removing the microphone artifact at the end of the recording.
        :returns: A list of Segment objects, each of which is one minute long at most (and only the last one - if any - will be less than one minute).
        """
        starts = range(0, int(round(self.duration_seconds * MS_PER_S)), MS_PER_MIN)
        stops = (min(self.duration_seconds * MS_PER_S, start + MS_PER_MIN) for start in starts)
        wav_outs = [self[start:stop] for start, stop in zip(starts, stops)]

        # Now cut out the last three seconds of the last item in wav_outs (it will just be microphone artifact)
        # or, if the last item is less than three seconds, just get rid of it
        if strip_last_seconds:
            if wav_outs[-1].duration_seconds > 3:
                wav_outs[-1] = wav_outs[-1][:-MS_PER_S * 3]
            else:
                wav_outs = wav_outs[:-1]

        return wav_outs

    def spectrogram(self, start_s=None, duration_s=None, start_sample=None, num_samples=None,
                    window_length_s=None, window_length_samples=None, overlap=0.5):
        """
        Does a series of FFTs from start_s or start_sample for duration_s or num_samples.
        Effectively, transforms a slice of the AudioSegment into the frequency domain across different
        time bins.

        :param start_s: The start time. Starts at the beginning if neither this nor start_sample is specified.
        :param duration_s: The duration of the spectrogram in seconds. Goes to the end if neither this nor
                           num_samples is specified.
        :param start_sample: The index of the first sample to use. Starts at the beginning if neither this nor
                             start_s is specified.
        :param num_samples: The number of samples in the spectrogram. Goes to the end if neither this nor
                            duration_s is specified.
        :param window_length_s: The length of each FFT in seconds. If the total number of samples in the spectrogram
                                is not a multiple of the window length in samples, the last window will be zero-padded.
        :param window_length_samples: The length of each FFT in number of samples. If the total number of samples in the
                                spectrogram is not a multiple of the window length in samples, the last window will
                                be zero-padded.
        :param overlap: The fraction of each window to overlap.
        :returns: Three np.ndarrays: The frequency values in Hz (the y-axis in a spectrogram), the time values starting
                  at start time and then increasing by duration_s each step (the x-axis in a spectrogram), and
                  the dB of each time/frequency bin as a 2D array of shape [len(frequency values), len(duration)].
        :raises ValueError: If start_s and start_sample are both specified, if duration_s and num_samples are both
                            specified, if the first window's duration plus start time lead to running off the end
                            of the AudioSegment, or if window_length_s and window_length_samples are either
                            both specified or if they are both not specified.
        """
        if start_s is not None and start_sample is not None:
            raise ValueError("Only one of start_s and start_sample may be specified.")
        if duration_s is not None and num_samples is not None:
            raise ValueError("Only one of duration_s and num_samples may be specified.")
        if window_length_s is not None and window_length_samples is not None:
            raise ValueError("Only one of window_length_s and window_length_samples may be specified.")
        if window_length_s is None and window_length_samples is None:
            raise ValueError("You must specify a window length, either in window_length_s or in window_length_samples.")

        if start_s is None and start_sample is None:
            start_sample = 0
        if duration_s is None and num_samples is None:
            num_samples = len(self.get_array_of_samples()) - int(start_sample)

        if duration_s is not None:
            num_samples = int(round(duration_s * self.frame_rate))
        if start_s is not None:
            start_sample = int(round(start_s * self.frame_rate))

        if window_length_s is not None:
            window_length_samples = int(round(window_length_s * self.frame_rate))

        if start_sample + num_samples > len(self.get_array_of_samples()):
            raise ValueError("The combination of start and duration will run off the end of the AudioSegment object.")

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

# Tests
if __name__ == "__main__":
    #Uncomment to test
    #import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print("For testing this module, USAGE:", sys.argv[0], os.sep.join("path to wave file.wav".split(' ')))
        exit(1)

    print("Reading in the wave file...")
    seg = from_file(sys.argv[1])

    print("Information:")
    print("Channels:", seg.channels)
    print("Bits per sample:", seg.sample_width * 8)
    print("Sampling frequency:", seg.frame_rate)
    print("Length:", seg.duration_seconds, "seconds")

    print("Doing FFT and plotting the histogram...")
    print("  |-> Computing the FFT...")
    hist_bins, hist_vals = seg.fft()
    hist_vals = abs(hist_vals) / len(hist_vals)
    print("  |-> Plotting...")
    hist_vals_for_plot = 10 * np.log10(hist_vals + 1e-9)
    plt.plot(hist_bins / 1000, hist_vals_for_plot, linewidth=0.02)
    plt.xlabel("kHz")
    plt.ylabel("dB")
    plt.show()

    print("Doing a spectrogram...")
    print("  |-> Computing overlapping FFTs...")
    hist_bins, times, amplitudes = seg.spectrogram(start_s=10, duration_s=1.2, window_length_s=0.03, overlap=0.5)
    hist_bins = hist_bins / 1000
    amplitudes = abs(amplitudes) / len(amplitudes)
    amplitudes = 10 * np.log10(amplitudes + 1e-9)
    print("  |-> Plotting...")
    x, y = np.mgrid[:len(times), :len(hist_bins)]
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, amplitudes)
    plt.show()

    print("Detecting voice...")
    seg = seg.resample(sample_rate_Hz=32000, sample_width=2, channels=1)
    results = seg.detect_voice()
    voiced = [tup[1] for tup in results if tup[0] == 'v']
    unvoiced = [tup[1] for tup in results if tup[0] == 'u']
    print("  |-> reducing voiced segments to a single wav file 'voiced.wav'")
    voiced_segment = voiced[0].reduce(voiced[1:])
    voiced_segment.export("voiced.wav", format="WAV")
    print("  |-> reducing unvoiced segments to a single wav file 'unvoiced.wav'")
    unvoiced_segment = unvoiced[0].reduce(unvoiced[1:])
    unvoiced_segment.export("unvoiced.wav", format="WAV")

    print("Splitting into frames...")
    segments = [s for s in seg.generate_frames_as_segments(frame_duration_ms=1000, zero_pad=True)]
    print("Got this many segments after splitting them up into one second frames:", len(segments))

    print("Removing silence from voiced...")
    seg = voiced_segment.filter_silence()
    outname_silence = "nosilence.wav"
    seg.export(outname_silence, format="wav")
    print("After removal:", outname_silence)

