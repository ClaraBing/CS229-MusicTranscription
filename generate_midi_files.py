import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet
import sys
from util import outputMIDI
debug = False
K=5

annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/')

# Load CNN results from validation set
filepath = "dataset/val_result_mtrx.npy"
validation_inference = np.load(filepath)

# Load trained Maximum Likelihood Estimates
trained_mle = "dataset/transitions_mle.npy" # Path to saves parameters for HMM
transitions = np.load(trained_mle).item()

# Run inference on each song
M = 300
offset = 0
threshold = 0.65
for songID in range(len(val_set.songNames)):
    total_solution = []
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
    print (np.sum(bins[i]>109))
    # Initialize CSP
    # Solve tracking problem independently on smaller portions and patch together
    patches = int(N / M) # number of patches
    remainder = N - patches * M
    if debug: print (patches, remainder)
    lastBin = 0
    if debug: print ("Pitch tracking on each fragment")
    sys.stdout.flush()
    for i in range(patches):
        # print ('Fragment %d to %d' % (i * M, (i + 1) * M))
        pitch_contour = PitchContour(threshold = threshold)
        pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
        pitch_contour.setStartProbability(lambda v : transitions[(lastBin, v)])
        # print ("Setting notes for the CSP")
        pitch_contour.setNotes(M, K, probabilities[i * M:(i + 1) * M, :], bins[i * M:(i + 1) * M, :])
        # print ("Solving CSP...")
        solution = pitch_contour.solve()
        for v, pitch in solution.items():
          total_solution.append(pitch)
        lastBin = solution[M-1]

    if remainder > 0:
        # print ('Fragment %d to %d' % (patches * M, N))
        pitch_contour = PitchContour()
        pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
        pitch_contour.setStartProbability(lambda v : transitions[(lastBin, v)])
        # print ("Setting notes for the CSP")
        pitch_contour.setNotes(remainder, K, probabilities[patches*M:N, :], bins[patches*M:N, :])
        # print ("Solving CSP...")
        solution = pitch_contour.solve()
        for v, pitch in solution.items():
          total_solution.append(pitch)
    outputMIDI(len(total_solution), total_solution, songName+"_inference", duration_sec=0.058, tempo=120)
    outputMIDI(len(val_set.pitches[songID]), val_set.pitches[songID], songName+"_original", duration_sec=0.058, tempo=120)
    print ("Saved midi files for song " + songName)
    sys.stdout.flush()
