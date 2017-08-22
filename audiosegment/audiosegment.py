"""
This module simply exposes a wrapper of a pydub.AudioSegment object.
"""
from __future__ import division

import collections
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
            construct_segment = lambda frames: AudioSegment(pydub.AudioSegment(data=b''.join([f.bytes for f in frames]),
                                                                          sample_width=self.sample_width,
                                                                          frame_rate=self.frame_rate,
                                                                          channels=self.channels), self.name)
            num_padding_frames = int(padding_duration_ms / frame_duration_ms)
            ring_buffer = collections.deque(maxlen=num_padding_frames)
            triggered = False
            voiced_frames = []
            for frame in frames:
                if not triggered:
                    ring_buffer.append(frame)
                    num_voiced = len([f for f in ring_buffer if v.is_speech(f.bytes, self.frame_rate)])
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                else:
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

    def generate_frames(self, frame_duration_ms, zero_pad=True):
        """
        Yields self's data in chunks of frame_duration_ms.

        This function adapted from pywebrtc's example [https://github.com/wiseman/py-webrtcvad/blob/master/example.py].

        :param frame_duration_ms: The length of each frame in ms.
        :param zero_pad: Whether or not to zero pad the end of the Segment object to get all the audio data out as frames. If not,
                         there may be a part at the end of the Segment that is cut off (the part will be <= frame_duration_ms in length).
        :returns: A Frame object with properties 'bytes (the data)', 'timestamp (start time)', and 'duration'.
        """
        Frame = collections.namedtuple("Frame", "bytes timestamp duration")

        bytes_per_frame = int(self.frame_rate * (frame_duration_ms / 1000) * self.sample_width)  # (samples/sec) * (seconds in a frame) * (bytes/sample)
        offset = 0  # where we are so far in self's data (in bytes)
        timestamp = 0.0  # where we are so far in self (in seconds)
        frame_duration_s = (bytes_per_frame / self.frame_rate) / self.sample_width  # (bytes/frame) * (sample/bytes) * (sec/samples)
        while offset + bytes_per_frame < len(self.raw_data):
            yield Frame(self.raw_data[offset:offset + bytes_per_frame], timestamp, frame_duration_s)
            timestamp += frame_duration_s
            offset += bytes_per_frame

        if zero_pad:
            rest = self.raw_data[offset:]
            zeros = bytes(bytes_per_frame - len(rest))
            yield Frame(rest + zeros, timestamp, frame_duration_s)

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

# Tests
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("For testing this module, USAGE:", sys.argv[0], os.sep.join("path to wave file.wav".split(' ')))
        exit(1)

    print("Reading in the wave file...")
    dubseg = pydub.AudioSegment.from_wav(sys.argv[1])
    seg = AudioSegment(dubseg, sys.argv[1])

    print("Information:")
    print("Channels:", seg.channels)
    print("Bits per sample:", seg.sample_width * 8)
    print("Sampling frequency:", seg.frame_rate)
    print("Length:", self.duration)

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

    print("Removing silence from voiced...")
    seg = voiced_segment.filter_silence()
    outname_silence = "nosilence.wav"
    seg.export(outname_silence, format="wav")
    print("After removal:", outname_silence)

