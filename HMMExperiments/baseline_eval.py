import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet
from util import getBinFromFrequency, read_melody
from librosa_baseline import baseline_tracking
import sys
annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
audio_folder = '/root/MedleyDB_selected/Audios/

def evaluation_error(gt, pred):
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
	err = mean((getBinFromFrequency(x) == getBinFromFrequency(y) for (x,y) in zip(pred_interval_avg, gt_interval_avg)))
	return err

baseline_acc = 0.0
for filename in os.listdir(annotations_val):
	if filename.endswith(".csv"):
		# The ordering/lengths of songs can be determined following the code below:
		audioName = filename[:filename.find('MELODY')-1]
		audio_file = audio_folder + audio_name + '/' + audio_name + '_MIX.wav'
		raw_pitch, processed_pitch = baseline_tracking(audio_file)
		_, gt = read_melody(audio_name)
		processed_pitch = np.asarray(processed_pitch)
		print(np.asarray(gt).shape, processed_pitch.shape)
		baseline_acc += evaluation_error(gt, processed_pitch)

print ("Baseline accuracy: ", baseline_acc / len(os.listdir(annotations_val)))
