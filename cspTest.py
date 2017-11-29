import unittest

from pitch_contour import *

class PitchContourTest(unittest.TestCase):

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

    def test_inference(self):
        flat_pitch = PitchContour()
        K = 2
        N = 4
        frequencies = [
        [9, 10],
        [1, 9],
        [1, 8],
        [5, 7]
        ]
        probabilities = [
        [0.1, 0.9],
        [0.1, 0.9],
        [0.5, 0.5],
        [0, 1]
        ]
        variances = [
        [0.01 for i in range(3)]
        for i in range(4)
        ]
        flat_pitch.setRange(range(1, 11))
        flat_pitch.setNotes(N, K, probabilities, frequencies, variances)
        solutionCSP = flat_pitch.solve()
        solution = {0 : 10, 1: 9, 2: 8, 3: 7}
        self.assertTrue(solution == solutionCSP)

    def test_both(self):
        data = [[1, 2 ,3, 2, 4], [1, 2, 4, 3, 4], [1, 1, 2]]
        transition = np.load("train.npy").item()
        pitch = PitchContour()
        K = 2
        N = 4
        frequencies = [
        [1, 4],
        [2, 3],
        [3, 1],
        [2, 4]
        ]
        probabilities = [
        [0.9, 0.1],
        [0.5, 0.5],
        [0.45, 0.55],
        [1, 0]
        ]
        variances = [
        [0.01 for i in range(3)]
        for i in range(4)
        ]
        pitch.setRange(range(5))
        pitch.setTransitionProbability(lambda f1, f2 : transition[(f1, f2)])
        pitch.setNotes(N, K, probabilities, frequencies, variances)
        solutionCSP = pitch.solve()
        solution = {0 : 1, 1: 2, 2: 3, 3: 2}
        self.assertTrue(solution == solutionCSP)

if __name__ == '__main__':
    unittest.main()
