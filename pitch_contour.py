import numpy as np
import math
import sys
import collections, util, copy
from csp import *

# Laplace distribution evaluation for a change of pitch period between two frames.
# Enforces temporal continuity between two adjacent states.
def laplaceDistribution(delta, mu=0.4, sigma=2.4):
    return 1.0 / (2 * sigma) * np.exp(-abs(delta - mu)/ sigma)

# Normal distribution evaluation for a given value, mean and standard deviation
def pdf(mean, std, value):
    u = float(value - mean) / abs(std)
    y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
    return y

# Gaussian Mixture Model evaluation for a state given the K most probables frequencies
# observed by the CNN.
# Input: Integer K,
# probabilities: array of size K of the normalized probabilities for the K most probable frequencies
# given by the CNN.
# frequencies: array of size K with the values of the frequencies
# variances: array of size K with the values of the standard deviation
def gaussianMixtureModelDistribution(observation, K, probabilities, frequencies, variances):
    normals = [pdf(frequencies[i], variances[i], observation) for i in range(K)]
    return np.sum(np.multiply(normals, probabilities))

# Default transition probability function between two frequencies
def transition_probability(fbefore, fafter):
    if fbefore > 0 and fafter > 0:
        return laplaceDistribution(abs(1.0 / fbefore- 1.0 / fafter))
    else:
        return 1.0

# Get bin number from frequency using formula
# n = floor(log(2^(57/12)*f/f0)/ log(\sqrt[12]{2})
# why adding 2^(57/12): to make the output non-negative
def getBinFromFrequency(frequency, base = 440.0):
    if frequency == 0.0:
        return 0
    else:
        return round((math.log(frequency/base) / math.log(math.pow(2.0, 1/ 12.0)))) + 58

# Get frequency from bin using the formula
def getFrequencyFromBin(bin, base = 440.0):
    if bin == 0:
        return 0.0
    else:
	    return base * math.pow(2.0, (bin - 58) / 12.0)

class PitchContour(CSP):
    def __init__(self):
        super(PitchContour, self).__init__()
        self.probStart = None
        self.probTrans = None
        self.probEmission = None

    # Set ranges of frequencies that are possible.
    def setRange(self, ranges):
        self.range = ranges

    # Set the start probability of the start state
    def setStartProbability(self, probability):
        self.probStart = probability

    # Set the transition probability between two hidden states
    def setTransitionProbability(self, prob):
        self.probTrans = prob

    # Set the emission probability between a hidden state and the observation
    def setEmissionProbability(self, prob):
        self.probEmission = prob

    # Set number of notes to be determined in the pitch contour.
    # Probabilities is a N x K matrix, each line represents the probabilities for
    # the K most probable pitches for the note.
    # Frequencies is N x K
    # Variances is N x K
    def setNotes(self, N, K, probabilities, frequencies, variances):
        # Set emission probability to default if nothing was specified
        if self.probEmission is None:
            probEmission = lambda i : lambda v : gaussianMixtureModelDistribution(v, K, \
                probabilities[i], frequencies[i], variances[i])
        else:
            probEmission = self.probEmission
        # Set transition probability to laplacian model is nothin was specified
        if self.probTrans is None:
            probTrans = transition_probability
        else:
            probTrans = self.probTrans
        # Set start probability to uniform if nothing was specified
        if self.probStart is None:
            probStart = lambda v : 1.0 / len(self.range)
        else:
            probStart = self.probStart

        for i in range(N):
            self.add_variable(i, self.range)
            self.add_unary_factor(i, probEmission(i))
            if i > 0:
                self.add_binary_factor(i-1, i, probTrans)
            else:
                self.add_unary_factor(i, probStart)

    def solve(self):
        search = BacktrackingSearch()
        search.solve(self)
        search.print_stats()
        self.solutions = search.optimalAssignment
        return self.solutions

    def print_solution(self):
        print (self.solutions)

# Learn the transition probability from the data with laplace smoothing of parameter alpha
# Data contains N lines that contains the sequence of frequencies.
# Perform learning by counting each transitions and normalizing
# Save the transitions in file
def trainTransition(data, frequencies,  outputFile, alpha=1):
    # Dictionary containing all values of function (fbefore, fafter) -> probability
    transitionProb = collections.defaultdict(float)
    # Dictionary containing counts of function (fbefore)
    counts = collections.defaultdict(int)
    for i in frequencies:
        for j in frequencies:
            transitionProb[(i,j)] = alpha
        counts[i] += len(frequencies) * alpha
    # Count
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            transitionProb[(data[i][j-1], data[i][j])] += 1
            counts[data[i][j-1]] += 1
    # Normalize
    for (i,j), _ in transitionProb.items():
        if counts[i] > 0:
            transitionProb[(i,j)] *= (1.0 / counts[i])
    np.save(outputFile, transitionProb)
