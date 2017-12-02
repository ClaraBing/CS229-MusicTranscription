import unittest

from pitch_contour import *
import numpy as np

np.random.seed(10)
class PitchContourTest(unittest.TestCase):
    def enumerate_assignments(i, assignment, N, pitch, frequencies):
        if i == N:
            print (pitch.get_assignment_weight(assignment), assignment)
        else:
            for j in frequencies[i]:
                assignment[i] = j
                enumerate_assignments(i + 1, assignment, N, pitch, frequencies)

    def test_freqbin(self):
        bin = getBinFromFrequency(440.0)
        self.assertEqual(bin, 58)
        self.assertEqual(getBinFromFrequency(523), 61)

    def test_training(self):
        data = [[1, 2 ,3, 2, 4], [1, 2, 4, 3, 4], [1, 1, 2]]
        trainTransition(data, range(5), "train.npy")
        probabilities = np.load("train.npy").item()
        # Verify that all probabilities sum to 1.
        for i in range(5):
            count = 0
            for j in range(5):
                count += probabilities[(i,j)]
            self.assertEqual(int(round(count)), 1)

    def test_get_weights(self):
        flat_pitch = PitchContour()
        K = 2
        N = 4
        frequencies = np.array([
        [16, 15],
        [15, 14],
        [12, 13],
        [10, 12]
        ])
        probabilities = np.array([
        [0.1, 0.9],
        [0.1, 0.9],
        [0.45, 0.55],
        [0, 1]
        ])
        flat_pitch.setNotes(N, K, probabilities, frequencies)

        assignment = {0: 15, 1: 14, 2: 13, 3: 12}
        self.assertEqual(flat_pitch.get_assignment_weight(assignment), \
            flat_pitch.get_delta_weight({}, 0, 15) + flat_pitch.get_delta_weight({0:15}, 1, 14) + \
            flat_pitch.get_delta_weight({0:15, 1: 14}, 2, 13) + flat_pitch.get_delta_weight({0:15, 1: 14, 2: 13}, 3, 12))

    def test_inference(self):
        flat_pitch = PitchContour()
        K = 2
        N = 4
        frequencies = np.array([
        [16, 15],
        [15, 14],
        [12, 13],
        [10, 12]
        ])
        probabilities = np.array([
        [0.1, 0.9],
        [0.1, 0.9],
        [0.45, 0.55],
        [0, 1]
        ])
        flat_pitch.setNotes(N, K, probabilities, frequencies)
        solutionCSP = flat_pitch.solve()
        solutionBacktrack = flat_pitch.solve(mode='backtrack')
        self.assertTrue(solutionBacktrack == solutionCSP)

    def test_both(self):
        data = [[1, 2 ,3, 2, 4], [1, 2, 4, 3, 4], [1, 1, 2]]
        transition = np.load("train.npy").item()
        pitch = PitchContour()
        K = 2
        N = 4
        bins = np.array([
        [1, 4],
        [2, 3],
        [3, 1],
        [2, 4]
        ])
        probabilities = np.array([
        [0.9, 0.1],
        [0.5, 0.5],
        [0.45, 0.55],
        [1, 0]
        ])
        pitch.setTransitionProbability(lambda f1, f2 : transition[(f1, f2)])
        pitch.setNotes(N, K, probabilities, bins)
        solutionCSP = pitch.solve()
        solutionBacktrack = pitch.solve(mode='backtrack')
        print (solutionBacktrack, solutionCSP)

    def test_gibbs(self):
        data = [[1, 2 ,3, 2, 4], [1, 2, 4, 3, 4], [1, 1, 2]]
        transition = np.load("train.npy").item()
        pitch = PitchContour()
        K = 2
        N = 4
        bins = np.array([
        [1, 4],
        [2, 3],
        [3, 1],
        [2, 4]
        ])
        probabilities = np.array([
        [0.9, 0.12],
        [0.5, 0.5],
        [0.45, 0.55],
        [1, 0]
        ])
        pitch.setTransitionProbability(lambda f1, f2 : transition[(f1, f2)])
        pitch.setNotes(N, K, probabilities, bins)
        solutionCSP = pitch.solve(mode="gibbs")
        solutionBacktrack = pitch.solve(mode='backtrack')
        print (solutionBacktrack, solutionCSP)

if __name__ == '__main__':
    unittest.main()
