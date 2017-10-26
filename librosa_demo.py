import matplotlib
# import IPython.display
import numpy as np
import librosa
import time

# y, sr = librosa.load(librosa.util.example_audio_file(), duration=5.0)
start = time.clock()
audio_file = 'LizNelson_Rainfall_MIX.wav'
y, sr = librosa.load(audio_file)
print('Audio file loaded: ' + audio_file)
print('{:f}s for loading the audio file.'.format(time.clock()-start))

start = time.clock()
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
print('{:f}s for piptrack'.format(time.clock()-start))
print("len(pitches): {:d}".format(len(pitches)))
print("pitch shape:")
print(pitches.shape)
d_range, time_range = pitches.shape
for t in range(time_range):
	pitch_t = pitches[:,t]
	pitch_t = at_t[np.nonzero(pitch_t)]
	print(t)
	print(pitch_t)
	print()
# print(any(pitches))
print('\n\nlen(magnitudes): {:d}'.format(len(magnitudes)))
# print(any(magnitudes))
# print(magnitudes)