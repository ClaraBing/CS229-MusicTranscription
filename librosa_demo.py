import matplotlib
# import IPython.display
import numpy as np
import librosa
import time
import matplotlib.pylab as plt

debug = False
database_sr = 44100

def main():
	audio_name = 'LizNelson_Rainfall_MIX'
	audio_file = 'data/Audio/LizNelson_Rainfall/' + audio_name + '.wav'
	result_file = 'data/' + audio_name + '.csv'
	raw_pitch, processed_pitch = baseline_tracking(audio_file, result_file)
	pitch_plot(raw_pitch) # pass in input as needed


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
	# Xiaoyan: please modify this function
	# x, y = [], []
	# for i, notes_grp in enumerate(pitches):
	# 	x.extend([i] * len(notes_grp))
	# 	y.extend(notes_grp)
	# plt.plot(x, y, label='time step (x) vs frequency (y)')


	def plot(vector, name, xlabel=None, ylabel=None):
	    plt.figure()
	    plt.plot(vector)
	    plt.xlabel(xlabel)
	    plt.ylabel(ylabel)
	    plt.plot()
	    plt.savefig('pitch_plots/'+name)

	def smooth(x,window_len=11,window='hanning'):
	        if window_len<3:
	                return x
	        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
	                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
	        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
	        if window == 'flat': #moving average
	                w=np.ones(window_len,'d')
	        else:
	                w=eval('np.'+window+'(window_len)')
	        y=np.convolve(w/w.sum(),s,mode='same')
	        return y[window_len:-window_len+1]

	def extract_max_plot(pitches, shape):
	    new_pitches = []
	    for i in range(0, shape[1]):
	        new_pitches.append(np.max(pitches[:,i]))
	    return new_pitches


	shape = np.shape(pitches)

	pitches = extract_max_plot(pitches, shape)

	pitches1 = smooth(pitches,window_len=10)
	pitches2 = smooth(pitches,window_len=20)
	pitches3 = smooth(pitches,window_len=30)
	pitches4 = smooth(pitches,window_len=40)

	plot(pitches1, 'pitches1')
	plot(pitches2, 'pitches2')
	plot(pitches3, 'pitches3')
	plot(pitches4, 'pitches4')
	plot(pitches, 'pitches')

main()
