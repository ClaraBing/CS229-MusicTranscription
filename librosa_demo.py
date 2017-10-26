import matplotlib
# import IPython.display
import numpy as np
import librosa
import time

debug = False
database_sr = 44100

def main():
	audio_name = 'LizNelson_Rainfall_MIX'
	audio_file = 'data/Audio/LizNelson_Rainfall/' + audio_name + '.wav'
	result_file = 'data/' + audio_name + '.txt'
	raw_pitch, processed_pitch = baseline_tracking(audio_file, result_file)
	# pitch_plot() # pass in input as needed


def extract_pitch_max(pitches, magnitudes, timerange):
	new_pitches = []
	new_magnitudes = []
	for i in range(timerange):
		maxMagn = max(magnitudes[:,i])
		index = np.argmax(magnitudes[:,i])
		new_pitches.append(pitches[index,i])
		new_magnitudes.append(maxMagn)
	return (new_pitches,new_magnitudes)

# Takes input file and result file
# Outputs raw pitches from the input file,
# and processed pitches (without 0 values and with only max value for each timestamp)
# Outputs processed pitches in result_file
def baseline_tracking(audio_file, result_file=None):
	start = time.clock()
	y, sr = librosa.load(audio_file)
	print('Audio file loaded: ' + audio_file)
	print('{:f}s for loading the audio file.'.format(time.clock()-start))

	start = time.clock()
	pitches, magnitudes = librosa.piptrack(y=y, sr=database_sr)
	mag_thresh = 2*np.mean(magnitudes)/3
	d_range, time_range = pitches.shape
	pitches_max, magnitudes_max = \
		extract_pitch_max(pitches, magnitudes, time_range)
	ret_pitch = []
	if result_file:
		file = open(result_file, 'w+')
	# print out tracked notes over time
	for t in range(time_range):
		# filter out frequencies with zero value
		pitch_t = pitches_max[t]
		# filter out notes that are too weak
		mag_t = magnitudes_max[t]
		if debug:
			print("pitch_idx shape:")
			print(pitch_idx.shape)
			print("mag_idx shape:")
			print(mag_idx.shape)
			print("idx shape:")
			print(merged_idx.shape)

		# only print at a time t if there're notes present
		if pitch_t != 0 and mag_t > mag_thresh:
			ret_pitch.append(pitch_t)
			if result_file:
				file.write(str(t)+ ',' + str(pitch_t)+'\n')
			else:
				print(t)
				print(pitch_t)
				print()
	if result_file:
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
