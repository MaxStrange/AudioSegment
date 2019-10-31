import sys
sys.path.insert(0, '../')
import audiosegment as asg
import os

#testsuites
import casa
import fft
import filterbank
import human_audible
import nparray
import read_from_file
import resample
import serde
import silence
import spectrogram
import trim
import vad
import visualize

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE:", sys.argv[0], os.sep.join("path to wave file.wav".split(' ')))
        exit(1)

    os.makedirs("results", exist_ok=True)

    seg = read_from_file.test(sys.argv[1])

    # Print some information about the AudioSegment
    print("Information:")
    print(seg)
    print("Channels:", seg.channels)
    print("Bits per sample:", seg.sample_width * 8)
    print("Sampling frequency:", seg.frame_rate)
    print("Length:", seg.duration_seconds, "seconds")
    print("Fraction of audio that is human audible:", human_audible.test(seg) / (len(seg) / 1000))

    #casa.test(seg)  # Test takes too long, so you should really only run this one manually
    resampled = resample.test(seg)
    serde.test(resampled)
    nparray.test(resampled.resample(channels=2))
    slices = trim.test(resampled)
    fft.test(resampled)
    filterbank.test(resampled)
    spectrogram.test(resampled)
    silence.test(resampled)
    vad.test(resampled)
