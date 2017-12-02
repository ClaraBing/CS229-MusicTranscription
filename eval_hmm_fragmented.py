import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet


def getNumberOfHits(ground_truth, prediction, N):
    numCorrect = 0
    for i in range(N):
          if ground_truth[i] == prediction[i]:
            numCorrect +=1
    return numCorrect

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
totalAccuracy = 0
cnnOnlyAccuracy = 0
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
        print probabilities[i]
        bins[i] = validation_inference[i + offset][:K][:,1]
    offset += N
    # Initialize CSP
    # Solve tracking problem independently on smaller portions and patch together
    M = 100 # Size of portions
    patches = int(N / M) # number of patches
    remainder = N - patches * M
    notes = []
    lastBin = 0
    print ("Pitch tracking on each fragment")
    for i in range(patches):
        print ('Fragment %d to %d' % (i * M, (i + 1) * M))
        pitch_contour = PitchContour()
        pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
        pitch_contour.setStartProbability(lambda v : transitions[(lastBin, v)])
        print ("Setting notes for the CSP")
        pitch_contour.setNotes(M, K, probabilities[i * M:(i + 1) * M, :], bins[i * M:(i + 1) * M, :])
        print ("Solving CSP...")
        solution = pitch_contour.solve()
        for j in range(M):
            notes.append(solution[j])
        lastBin = solution[M-1]
        currentAccuracy = getNumberOfHits(val_set.pitches[songID][i*M:(i+1)* M], solution, M)
        currentCnnOnlyAccuracy = getNumberOfHits(val_set.pitches[songID][i*M:(i+1)* M], bins[:, 0][i*M:(i+1)* M], M)
        print ("With HMM: Accuracy rate on this song %f " % (currentAccuracy/M))
        print ("Without HMM: Accuracy rate on this song %f " % (currentCnnOnlyAccuracy/M))
        cnnOnlyAccuracy += currentCnnOnlyAccuracy
        totalAccuracy += currentAccuracy

    if remainder > 0:
        print ('Fragment %d to %d' % (N * M, N))
        pitch_contour = PitchContour()
        pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
        pitch_contour.setStartProbability(lambda v : transitions[(lastBin, v)])
        print ("Setting notes for the CSP")
        pitch_contour.setNotes(M, K, probabilities[N*M:N, :], bins[N*M:N, :])
        print ("Solving CSP...")
        solution = pitch_contour.solve()
        for key, v in solution.items():
            notes.append(v)
        currentAccuracy = getNumberOfHits(val_set.pitches[songID][N*M:N], solution, remainder)
        currentCnnOnlyAccuracy = getNumberOfHits(val_set.pitches[songID][N*M:N], bins[:, 0][N*M:N], remainder)
        print ("With HMM: Accuracy rate on this song %f " % (currentAccuracy/remainder))
        print ("Without HMM: Accuracy rate on this song %f " % (currentCnnOnlyAccuracy/remainder))
        cnnOnlyAccuracy += currentCnnOnlyAccuracy
        totalAccuracy += currentAccuracy

print ("With HMM: Total accuracy rate")
print (totalAccuracy/val_set.lengths[-1])
print ("Without HMM: Total accuracy rate")
print (cnnOnlyAccuracy/val_set.lengths[-1])
