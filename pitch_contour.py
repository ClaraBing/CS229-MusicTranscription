import numpy as np
import math
import sys
import collections, util, copy
from csp import *
from util import getBinFromFrequency, getFrequencyFromBin

# Laplace distribution evaluation for a change of pitch period between two frames.
# Enforces temporal continuity between two adjacent states.
def laplaceDistribution(delta, mu=0.4, sigma=2.4):
    return 1.0 / (2 * sigma) * np.exp(-abs(delta - mu)/ sigma)

# Normal distribution evaluation for a given value, mean and standard deviation
def pdf(mean, std, value):
    u = float(value - mean) / abs(std)
    y = (1.0 / (math.sqrt(2 * math.pi) * abs(std))) * math.exp(-u * u / 2.0)
    return y

# Multinomial distribution function for a given value, probability vector and values
def multinomial(value, probabilities, values):
    if value in values:
        return probabilities[values.tolist().index(value)]
    else:
        return 0.0

# Gaussian Mixture Model evaluation for a state given the K most probables frequencies
# observed by the CNN.
# Input: Integer K,
# probabilities: array of size K of the normalized probabilities for the K most probable frequencies
# given by the CNN.
# bins: array of size K with the values of possible bins
# variance is defaulted to 0.5 as the bins are integers.
def gaussianMixtureModelDistribution(observation, probabilities, bins):
    K = len(bins)
    normals = [pdf(bins[i], 0.5, observation) for i in range(K)]
    return np.sum(np.multiply(normals, probabilities))

# Default transition probability function between two bins
def transition_probability(binbefore, binafter, mu, sigma):
    fbefore = getFrequencyFromBin(binbefore)
    fafter = getFrequencyFromBin(binafter)
    if fbefore > 0 and fafter > 0:
        return laplaceDistribution(abs(1.0 / fbefore- 1.0 / fafter), mu, sigma)
    else:
        return 1.0 / 5 # Uniform distribution

class PitchContour(CSP):
    def __init__(self, emission='multinomial', transmission='mle', start='uniform', mu=0.4, sigma=2.4):
        super(PitchContour, self).__init__()
        self.probStart = None
        self.probTrans = None
        self.probEmission = None
        self.emissionMode = emission
        self.transmissionMode = transmission
        self.startMode = start
        self.mu = mu
        self.sigma = sigma

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
    # Bins is N x K and values are between 0 and 108.
    def setNotes(self, N, K, probabilities, bins):
        # Set emission probability to default if nothing was specified
        if self.probEmission is None:
            if self.emissionMode == 'multinomial':
                probEmission = lambda i : lambda v : multinomial(v, probabilities[i], bins[i])
            elif self.emissionMode == 'gaussian':
                probEmission = lambda i : lambda v : \
                    gaussianMixtureModelDistribution(v, probabilities[i], bins[i])
        else:
            probEmission = self.probEmission
        # Set transition probability to laplacian model if nothing was specified
        if self.probTrans is None:
            probTrans = lambda u,v : transition_probability(u, v, self.mu, self.sigma)
        else:
            probTrans = self.probTrans
        # Set start probability to uniform if nothing was specified
        if self.probStart is None:
            if self.startMode == 'uniform':
                probStart = lambda v : 1.0 / 109
            else:
                probStart = lambda v : self.probTrans(0, v)
        else:
            probStart = self.probStart

        for i in range(N):
            # Add variable and constraint that variable is in the set of bins
            if probabilities[i][0] > 0.8:
              self.add_variable(i, [bins[i][0]])
            else:
              self.add_variable(i, bins[i])
            self.add_unary_factor(i, probEmission(i))
            if i > 0:
                self.add_binary_factor(i-1, i, probTrans)
            else:
                self.add_unary_factor(i, probStart)

    def solve(self, mode = "gibbs"):
        if mode == "backtrack":
            search = BacktrackingSearch()
            search.solve(self)
            search.print_stats()
            self.solutions = search.optimalAssignment
        if mode == "gibbs":
            gibbs = GibbsSampling()
            self.solutions = gibbs.solve(self)
        return self.solutions

    def print_solution(self):
        print (self.solutions)

# Learn the transition probability from the data with laplace smoothing of parameter alpha
# Data contains N lines that contains the sequence of bins.
# Perform learning by counting each transitions and normalizing
# Save the transitions in file
def trainTransition(data, bins, outputFile, alpha=1):
    # Dictionary containing all values of function (fbefore, fafter) -> probability
    transitionProb = collections.defaultdict(float)
    # Dictionary containing counts of function (fbefore)
    counts = collections.defaultdict(int)

    for i in bins:
        for j in bins:
            transitionProb[(i,j)] = alpha
        counts[i] += len(bins) * alpha
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
