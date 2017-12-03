import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet
from util import getBinFromFrequency, read_melody
from librosa_baseline import baseline_tracking
import sys, os
from time import time
annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
audio_folder = '/root/MedleyDB_selected/Audios/'

def evaluation_acc(gt, pred):
	accurate = 0
	for i in range(len(gt)):
		accurate += getBinFromFrequency(pred[i]) == getBinFromFrequency(gt[i])
	return accurate / len(gt)

baseline_acc = []
sid = 0 # song id
total_start = time()
for filename in os.listdir(annotations_val):
	if filename.endswith(".csv"):
		start = time()
		sid += 1
		# The ordering/lengths of songs can be determined following the code below:
		audio_name = filename[:filename.find('MELODY')-1]
		audio_file = audio_folder + audio_name + '/' + audio_name + '_MIX.wav'
		print ('#', sid, " Baseline tracking for %s" % audio_name)
		raw_pitch, processed_pitch = baseline_tracking(audio_file)
		_, gt = read_melody(audio_name, sr_ratio = 2)
		if abs(len(gt)-len(processed_pitch))>1:
			raise ValueError('Dimension mismatch: gt={:d} / processed={:d}'.format(len(gt), len(processed_pitch)))
		min_len = min(len(gt), len(processed_pitch))
		processed_pitch, gt = processed_pitch[:min_len], gt[:min_len]
		processed_pitch = np.asarray(processed_pitch)
		acc = evaluation_acc(gt, processed_pitch)
		print('accuracy: ', acc)
		print('time: ', time()-start, '\n')
		baseline_acc += acc,
		sys.stdout.flush()

print("\nAccuracy Overview:")
print("Mean: ", sum(baseline_acc) / len(baseline_acc))
print("Median: ", sorted(baseline_acc)[int(len(baseline_acc)/2)])
print("Max: ", max(baseline_acc))
print("Min: ", min(baseline_acc))
print("=== Time: {:f}s * {:d} songs ===".format((time()-total_start)/len(baseline_acc), len(baseline_acc)))
