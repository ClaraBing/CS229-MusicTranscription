import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# our code
from PitchEstimationDataSet import PitchEstimationDataSet
import util

def plot_confusion_matrix(cm, classes=[],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cf_mtrx.png')


val_result = np.load('dataset/val_result_mtrx.npy')
annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/', sr_ratio = 2, audio_type='MIX')

val_pitches = []
for i in range(val_result.shape[0]):
	pitch_frame = val_result[i]
	max = -1
	max_pitch = 0
	for j in range(val_result.shape[1]):
		if (pitch_frame[j][0]>max):
			max = pitch_frame[j][0]
			max_pitch = pitch_frame[j][1]
	val_pitches.append(max_pitch)
val_pitches = np.asarray(val_pitches)

val_labels = []
for pitches in val_set.pitches:
	val_labels += pitches
val_labels = np.asarray(val_labels)

sampled_val_pitches = util.subsample(val_pitches, val_labels)

labels = range(109)
cnf_matrix = confusion_matrix(val_labels, sampled_val_pitches, labels = labels)
plot_confusion_matrix(cnf_matrix, title = 'cnf matrix')
