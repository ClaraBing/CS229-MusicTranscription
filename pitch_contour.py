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
def gaussianDistribution(x, mu, sigma):
    return 1.0 / math.sqrt(2 * math.pi * sigma ** 2) * \
        math.exp(-1.0/ (2 * sigma**2) * (x-mu)**2)

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

class PitchContour(CSP):
    def __init__(self):
        super(PitchContour, self).__init__()

    def setRange(self, ranges):
        self.range = ranges
    # Set number of notes to be determined in the pitch contour.
    # Probabilities is a N x K matrix, each line represents the probabilities for
    # the K most probable pitches for the note.
    # Frequencies is N x K
    # Variances is N x K
    def setNotes(self, N, K, probabilities, frequencies, variances):
        for i in range(N):
            self.add_variable(i, self.range)
            self.add_unary_factor(i, \
                lambda v : gaussianMixtureModelDistribution(v, K, \
                    probabilities[i], frequencies[i], variances[i]))
            if i > 0:
                self.add_binary_factor(i, i-1, \
                    lambda vafter, vbefore: laplaceDistribution(\
                        abs(1.0 / vafter- 1.0 / vbefore)) if vafter > 0 and vbefore > 0 else 1.0)

    def solve(self):
        search = BacktrackingSearch()
        search.solve(self)
        search.print_stats()
        self.solutions = search.optimalAssignment

    def print_solution(self):
        print self.solutions
