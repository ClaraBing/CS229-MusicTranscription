from csp import PitchContour, BacktrackingSearch

def construction_test():
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
    flat_pitch.solve()
    flat_pitch.print_solution()

construction_test()
