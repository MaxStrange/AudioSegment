# AudioSegment
Wrapper for pydub AudioSegment objects

## Example Usage
```python
import pydub
import audiosegment

print("Reading in the wave file...")
seg = audiosegment.from_file("whatever.wav")

print("Information:")
print("Channels:", seg.channels)
print("Bits per sample:", seg.sample_width * 8)
print("Sampling frequency:", seg.frame_rate)
print("Length:", seg.duration_seconds, "seconds")

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
```

## Notes
I may not get around to making the dependency on pydub invisible. Obviously it would be great if
it were the case, so that you don't have to include pydub and first make a pydub.AudioSegment
everytime we want to make an audiosegment.AudioSegment object, but I don't have time to think
about how to do it.

There is a hidden dependency on the command line program 'sox'. Pip will not install it for you.
You will have to install sox by:
- Debian/Ubuntu: `sudo apt-get install sox`
- Mac OS X: `brew install sox`
- Windows: `choco install sox`
