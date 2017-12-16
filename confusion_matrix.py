import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    ax=plt.gca() #get the current axes
#    PCM=ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
#    plt.colorbar(PCM, ax=ax) 
    #plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)

#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")

#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.save('cnf.png')

"""
test_result = np.load('dataset/test_result_mtrx.npy')
#print(test_result.shape)
annotations_test = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/test/'
test_set = PitchEstimationDataSet(annotations_test, '/root/data/test/', sr_ratio = 2)
"""
val_result = np.load('dataset/val_result_mtrx.npy')
annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/', sr_ratio = 2, audio_type='MIX')
print(val_result.shape)

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
print(val_pitches.shape)

val_labels = []
for pitches in val_set.pitches:
	val_labels += pitches
val_labels = np.asarray(val_labels)
print(val_labels.shape)

sampled_val_pitches = util.subsample(val_pitches, val_labels)
print(sampled_val_pitches.shape)

labels = range(109)
cnf_matrix = confusion_matrix(val_labels, sampled_val_pitches, labels = labels)
print(cnf_matrix.shape)
# np.save('cnf_matrix.npy',cnf_matrix)
plot_confusion_matrix(cnf_matrix[20:80, 0:80], title = 'cnf matrix')

# plt.save('cnf_try.png',)
