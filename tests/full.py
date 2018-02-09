import audiosegment as asg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

#testsuites
import casa
import normalize
import read_from_file
import resample
import trim
import visualize

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE:", sys.argv[0], os.sep.join("path to wave file.wav".split(' ')))
        exit(1)

    seg = read_from_file.test(sys.argv[1])

    # Print some information about the AudioSegment
    print("Information:")
    print("Channels:", seg.channels)
    print("Bits per sample:", seg.sample_width * 8)
    print("Sampling frequency:", seg.frame_rate)
    print("Length:", seg.duration_seconds, "seconds")

    resampled = resample.test(seg)

    #casa.test()

    normalized = normalize.test(resampled)

    slices = trim.test(resampled)



    exit()





    print("Doing FFT and plotting the histogram...")
    print("  |-> Computing the FFT...")
    hist_bins, hist_vals = seg[1:3000].fft()
    hist_vals = np.abs(hist_vals) / len(hist_vals)
    print("  |-> Plotting...")
#    hist_vals = 10 * np.log10(hist_vals + 1e-9)
    plt.plot(hist_bins / 1000, hist_vals)#, linewidth=0.02)
    plt.xlabel("kHz")
    plt.ylabel("dB")
    plt.show()

    print("Doing a spectrogram...")
    print("  |-> Computing overlapping FFTs...")
    hist_bins, times, amplitudes = seg[1:VISUALIZE_LENGTH].spectrogram(window_length_s=0.03, overlap=0.5)
    hist_bins = hist_bins / 1000
    amplitudes = np.abs(amplitudes) / len(amplitudes)
    amplitudes = 10 * np.log10(amplitudes + 1e-9)
    print("  |-> Plotting...")
    x, y = np.mgrid[:len(times), :len(hist_bins)]
    fig, ax = plt.subplots()
    ax.pcolormesh(x, y, amplitudes)
    plt.show()

    print("Removing silence...")
    seg = seg.filter_silence()
    outname_silence = "nosilence.wav"
    seg.export(outname_silence, format="wav")
    visualize(seg[:min(VISUALIZE_LENGTH, len(seg))], title="After Silence Removal")
    print("After removal:", outname_silence)

    print("Detecting voice...")
    results = seg.detect_voice(prob_detect_voice=0.7)
    voiced = [tup[1] for tup in results if tup[0] == 'v']
    unvoiced = [tup[1] for tup in results if tup[0] == 'u']
    print("  |-> reducing voiced segments to a single wav file 'voiced.wav'")
    if len(voiced) > 1:
        voiced_segment = voiced[0].reduce(voiced[1:])
    elif len(voiced) > 0:
        voiced_segment = voiced[0]
    else:
        voiced_segment = None
    if voiced_segment is not None:
        voiced_segment.export("voiced.wav", format="WAV")
        visualize(voiced_segment[:min(VISUALIZE_LENGTH, len(voiced_segment))], title="Voiced Segment")
    print("  |-> reducing unvoiced segments to a single wav file 'unvoiced.wav'")
    if len(unvoiced) > 1:
        unvoiced_segment = unvoiced[0].reduce(unvoiced[1:])
    elif len(unvoiced) > 0:
        unvoiced_segment = unvoiced[0]
    else:
        unvoiced_segment = None
    if unvoiced_segment is not None:
        unvoiced_segment.export("unvoiced.wav", format="WAV")
        visualize(unvoiced_segment[:min(VISUALIZE_LENGTH, len(unvoiced_segment))], title="Unvoiced Segment")

    print("Splitting into frames...")
    segments = [s for s in seg.generate_frames_as_segments(frame_duration_ms=1000, zero_pad=True)]
    print("Got this many segments after splitting them up into one second frames:", len(segments))

