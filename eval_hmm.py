import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet
from util import eval_accuracy
import sys # for flushing stdout buffer
from time import time


def getNumberOfHits(ground_truth, prediction, N):
    numCorrect = 0
    for i in range(N):
          if ground_truth[i] == prediction[i]:
            numCorrect +=1
    return numCorrect

annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/', sr_ratio=1, audio_type="MIX")

# Load CNN results from validation set
filepath = "dataset/val_result_mtrx.npy"
validation_inference = np.load(filepath)

# Load trained Maximum Likelihood Estimates
trained_mle = "dataset/transitions_mle.npy" # Path to saves parameters for HMM
transitions = np.load(trained_mle).item()
# Run inference on each song
K = 5
offset = 0
totalAccuracy = 0
cnnOnlyAccuracy = 0
start = time()
for songID in range(val_set.numberOfSongs):
    songName = val_set.songNames[songID]
    print ("Evaluating for " + songName)
    N = val_set.songLengths[songID]
    probabilities = np.zeros((N, K))
    bins = np.zeros((N,K))
    print ("Loading for %d notes" % N)
    for i in range(N):
        probabilities[i] = validation_inference[i + offset][:K][:,0]
        probabilities[i] /= np.sum(probabilities[i])
        bins[i] = validation_inference[i + offset][:K][:,1]
    offset += N
    # Initialize CSP
    pitch_contour = PitchContour()
    pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
    print ("Setting notes for the CSP")
    pitch_contour.setNotes(N, K, probabilities, bins)
    print ("Solving CSP...")
    solution = pitch_contour.solve()
    # currentAccuracy = getNumberOfHits(val_set.pitches[songID], solution, N)
    currentAccuracy = getNumberOfHits(val_set.pitches[songID], solution, N)
    currentCnnOnlyAccuracy = getNumberOfHits(val_set.pitches[songID], bins[:, 0], N)
    print ("With HMM: Accuracy rate on this song {:f}".format(currentAccuracy/N))
    print ("Without HMM: Accuracy rate on this song %f " % (currentCnnOnlyAccuracy/N))
    cnnOnlyAccuracy += currentCnnOnlyAccuracy
    totalAccuracy += currentAccuracy
    sys.stdout.flush()
print ("With HMM: Total accuracy rate")
print (totalAccuracy/val_set.lengths[-1])
print ("Without HMM: Total accuracy rate")
print (cnnOnlyAccuracy/val_set.lengths[-1])
print ('Using {:f} seconds'.format(time()-start))
