import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet
import sys

debug = False
K=5
b_prob = np.zeros((1,K))
w_prob = np.zeros((1,K))

def getNumberOfHits(ground_truth, prediction, N, probs=None):
    global b_prob
    global w_prob
    numCorrect = 0
    for i in range(N):
          if ground_truth[i] == prediction[i]:
            if probs is not None:
              b_prob = np.append(b_prob, probs[i,:].reshape((1,K)), axis = 0)
            numCorrect +=1
          else:
            if probs is not None:
              w_prob = np.append(w_prob, probs[i,:].reshape((1,K)), axis = 0)
    return numCorrect


# Pitch tracking using Gibbs sampling on given notes
'''
    Input:
        N : number of notes
        K : number of possibilities for one note
        M:  length of a fragment M <= N
        probabilities: (N,K) matrix, probabilities of each possibilities
        bins: (N, K) matrix: possible bins for each notes
        transitions: dictionary of transitions for MLE solver
        threshold: threshold to use for the solver
    Output:
        array of length N
'''


def fragmented_solver(N, K, M, probabilities, bins,
        transitions=None, threshold=0.6):
    # Initialize CSP
    total_solution = []
    patches = int(N / M)   # number of patches
    remainder = N - patches * M
    if debug:
        print (patches, remainder)
    lastBin = 0
    if debug:
        print ("Pitch tracking on each fragment")
    sys.stdout.flush()
    for i in range(patches):
        pitch_contour = PitchContour(threshold=threshold)
        pitch_contour.setTransitionProbability(
            lambda b1, b2: transitions[(b1, b2)])
        pitch_contour.setStartProbability(
            lambda v: transitions[(lastBin, v)])
        pitch_contour.setNotes(M, K, probabilities[i * M:(i + 1) * M, :],
            bins[i * M:(i + 1) * M, :])
        solution = pitch_contour.solve()
        lastBin = solution[M-1]
        for _, v in solution.items():
            total_solution.append(v)

    if remainder > 0:
        pitch_contour = PitchContour(threshold=threshold)
        pitch_contour.setTransitionProbability(
            lambda b1, b2: transitions[(b1, b2)])
        pitch_contour.setStartProbability(
            lambda v: transitions[(lastBin, v)])
        pitch_contour.setNotes(remainder, K,
            probabilities[patches*M:N, :], bins[patches*M:N, :])
        solution = pitch_contour.solve()
        for _, v in solution.items():
            total_solution.append(v)
    return total_solution


annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/')

# Load CNN results from validation set
filepath = "dataset/val_result_mtrx.npy"
validation_inference = np.load(filepath)

# Load trained Maximum Likelihood Estimates
trained_mle = "dataset/transitions_mle.npy" # Path to saves parameters for HMM
transitions = np.load(trained_mle).item()

# Run inference on each song
hmmTotalAccuracy = []
# rangeM = [20, 50, 100, 200, 300, 500, 1000]
# for M in [50, 100, 200, 300, 400, 500]:
M = 300
rangeT=[0.60, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]
for threshold in rangeT:
    totalAccuracy = 0
    cnnOnlyAccuracy = 0
    offset = 0

    for songID in range(len(val_set.songNames)):
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
            solution = fragmented_solver(N, K, M, probabilities, bins,
                transitions=transitions, threshold=threshold)
            currentAccuracy = getNumberOfHits(val_set.pitches[songID],
                solution, N, probabilities)
            currentCnnOnlyAccuracy = getNumberOfHits(
                val_set.pitches[songID], bins[:, 0], N)
            print ("With HMM: Accuracy rate on this song %f " %
                (currentAccuracy/N))
            print ("Without HMM: Accuracy rate on this song %f " %
                (currentCnnOnlyAccuracy/N))
            cnnOnlyAccuracy += currentCnnOnlyAccuracy
            totalAccuracy += currentAccuracy
            sys.stdout.flush()
            hmmTotalAccuracy.append(totalAccuracy/val_set.lengths[-1])



print (rangeT, hmmTotalAccuracy)
# print ("With HMM: Total accuracy rate")
# print (totalAccuracy/val_set.lengths[-1])
print ("Without HMM: Total accuracy rate")
print (cnnOnlyAccuracy/val_set.lengths[-1])

# np.save("bad_bin_hmm", w_bin)
