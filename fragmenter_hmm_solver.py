import numpy as np
from pitch_contour import PitchContour
import sys

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

debug = True
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
        if debug:
          print ("Fragment %d out of %d" % (i, patches))
        pitch_contour = PitchContour(threshold=threshold)
        pitch_contour.setTransitionProbability(
            lambda b1, b2: transitions[(b1, b2)])
        pitch_contour.setStartProbability(
            lambda v: transitions[(lastBin, v)])
        pitch_contour.setNotes(M, K, probabilities[i * M:(i + 1) * M, :], bins[i * M:(i + 1) * M, :])
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
        pitch_contour.setNotes(remainder, K, probabilities[patches*M:N, :], bins[patches*M:N, :])
        solution = pitch_contour.solve()
        for _, v in solution.items():
            total_solution.append(v)
    return total_solution
