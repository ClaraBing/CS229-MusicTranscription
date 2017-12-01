import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet


def getErrorRate(ground_truth, prediction, N):
    numErrors = 0
    numCorrect = 0
    M = len(ground_truth)
    assert M == N
    for i in range(N):
          if ground_truth[i] != prediction[i]:
            numErrors += 1
          else:
            numCorrect +=1
    return numErrors, numCorrect

annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/')

# Load CNN results from validation set
filepath = "val.npy"
validation_inference = np.load(filepath)

# Load trained Maximum Likelihood Estimates
trained_mle = "transitions_mle.npy" # Path to saves parameters for HMM
transitions = np.load(trained_mle).item()

# Run inference on each song
K = 5
offset = 0
totalErrorRate = 0
totalAccuracy = 0
for songID in range(val_set.numberOfSongs):
    songName = val_set.songNames[songID]
    print ("Evaluating for " + songName)
    N = val_set.songLengths[songID]
    probabilities = np.zeros((N, K))
    bins = np.zeros((N,K))
    print ("Loading for %d notes" % N)
    for i in range(N):
        probabilities[i] = validation_inference[i + offset][:K][:,0]
        bins[i] = validation_inference[i + offset][:K][:,1]
    offset += N
    # Initialize CSP
    print (N)
    print (len(val_set.pitches[songID]))
    pitch_contour = PitchContour()
    pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
    print ("Setting notes for the CSP")
    pitch_contour.setNotes(N, K, probabilities, bins)
    print ("Solving CSP...")
    solution = pitch_contour.solve()
    currentErrorRate, currentAccuracy = getErrorRate(val_set.pitches[songID], solution, N)
    print ("Error rate on this song %d out of %d " % (currentErrorRate, N))
    totalErrorRate += currentErrorRate
    totalAccuracy += currentAccuracy
print ("Total error rate %d out of %d" % (totalErrorRate, val_set.lengths[-1]))
print ("Total accuracy rate %d out of %d" % (totalAccuracy, val_set.lengths[-1]))
print (totalAccuracy/val_set.lengths[-1])
