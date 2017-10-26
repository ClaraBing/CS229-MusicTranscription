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

main()