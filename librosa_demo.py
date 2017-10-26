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

def load_txt(f_name):
	fin = open(f_name, 'r')
	return [line.strip().split(',') for line in fin.readlines()]

def get_err(gt, pred):
	# Normalize time
	# find the number of time intervals
	gt_interval = []
	curr_interval = -1.000
	for i in range(len(gt)):
		gt[i][0] = round(gt[i][0], 3) # precision = miliseconds
		if gt[i][0] == curr_interval:
			gt_interval[-1] += gt[i][1],
		else:
			if gt[i][0] - curr_interval != 0.001:
				# if skipping over some intervals -> put placeholdes
				n_diff = (gt[i][0]-curr_interval) / 0.001
				for i in range(n_diff):
					gt_interval += [0],
					curr_interval += 0.001
			gt_interval += [gt[i][1]],
			curr_interval = gt[i][0]
	interval_cnt = len(gt_interval)
	# normalize pitch time
	step = int(len(pred)/interval_cnt)
	pred_interval_avg = []
	gt_interval_avg = []
	for i in range(interval_cnt):
		pred_notes = [note for notes in pred[i*step:(i+1)*step] for note in notes]
		if pred_notes == []:
			pred += 0,
		else:
			pred += sum(pred_notes) / len(pred_notes),
		gt_interval_avg += sum(gt_interval[i]) / len(gt_interval[i])
	if len(pred_interval_avg) != len(gt_interval_avg):
		raise ValueError("Ground truth & prediction dimension mismatch")
	err = sum((x-y)**2 for (x,y) in zip(pred_interval_avg, gt_interval_avg)) / len(pred_interval_avg)
	return err

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
