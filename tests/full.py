import sys
sys.path.insert(0, '../')
import audiosegment as asg
import os

#testsuites
import casa
import fft
import filterbank
import normalize
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

    seg = read_from_file.test(sys.argv[1])

    # Print some information about the AudioSegment
    print("Information:")
    print(seg)
    print("Channels:", seg.channels)
    print("Bits per sample:", seg.sample_width * 8)
    print("Sampling frequency:", seg.frame_rate)
    print("Length:", seg.duration_seconds, "seconds")

    resampled = resample.test(seg)
    #casa.test()
    normalized = normalize.test(resampled)
    serde.test(normalized)
    slices = trim.test(normalized)
    fft.test(normalized)
    filterbank.test(normalized)
    spectrogram.test(normalized)
    silence.test(normalized)
    vad.test(normalized)
