import matplotlib
# import IPython.display
import numpy as np
import librosa
import time

debug = False

def main():
	audio_name = 'LizNelson_Rainfall_MIX'
	audio_file = 'data/Audio/LizNelson_Rainfall/' + audio_name + '.wav'
	result_file = 'data/' + audio_name + '.txt'
	raw_pitch, processed_pitch = baseline_tracking(audio_file, result_file)
	pitch_plot() # pass in input as needed

def baseline_tracking(audio_file, result_file=None):
	start = time.clock()
	y, sr = librosa.load(audio_file)
	print('Audio file loaded: ' + audio_file)
	print('{:f}s for loading the audio file.'.format(time.clock()-start))

	start = time.clock()
	pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

	d_range, time_range = pitches.shape
	mag_thresh = 2*np.mean(magnitudes)/3
	ret_pitch = []
	# print out tracked notes over time
	for t in range(time_range):
		# filter out frequencies with zero value
		pitch_t = pitches[:,t]
		pitch_idx = pitch_t!=0
		# filter out notes that are too weak
		mag_t = magnitudes[:,t]
		mag_idx = mag_t>mag_thresh
		# apply both filters
		merged_idx = np.logical_and(pitch_idx, mag_idx)

		if debug:
			print("pitch_idx shape:")
			print(pitch_idx.shape)
			print("mag_idx shape:")
			print(mag_idx.shape)
			print("idx shape:")
			print(merged_idx.shape)
		
		pitch_t = pitch_t[np.logical_and(pitch_idx, mag_idx)]
		# only print at a time t if there're notes present
		if len(pitch_t):
			ret_pitch += pitch_t,
			if result_file:
				for pitch in pitch_t:
					file.write(str(t)+ ',' + str(pitch)+'\n')
			else:
				print(t)
				print(pitch_t)
				print()
	print("Saved in " + result_file)
	file.close()

	print("Stat:")
	print('{:f}s for piptrack'.format(time.clock()-start))
	print("len(pitches): {:d}".format(len(pitches)))
	print("pitch shape:")
	print(pitches.shape)
	print("magnitudes shape:")
	print(magnitudes.shape)
	print("sampling rates:")
	print(sr)

	return pitches, ret_pitch

def pitch_plot(pitches):
	pass
	# Xiaoyan: please modify this function
	# x, y = [], []
	# for i, notes_grp in enumerate(pitches):
	# 	x.extend([i] * len(notes_grp))
	# 	y.extend(notes_grp)
	# plt.plot(x, y, label='time step (x) vs frequency (y)')

main()