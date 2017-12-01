import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet

annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/')
val_loader = DataLoader(val_set, shuffle=False, **kwargs)

# Load CNN results from validation set
filepath = "val.npy"
validation_inference = np.load(filepath)

# Load trained Maximum Likelihood Estimates
trained_mle = "transitions_mle.npy" # Path to saves parameters for HMM
transitions = np.load(trained_mle).item()

# Run inference on each song
K = 5
offset = 0
N = 150
for songID in val_set.numberOfSongs:
    songName = val_set.songNames
    N = val_set.songLengths[songID]
    probabilities = np.zeros((N, K))
    bins = np.zeros((N,K))
    for i in range(N):
        probabilities[i] = validation_inference[i + offset][:K][:,0]
        bins[i] = validation_inference[i + offset][:K][:,1]
    offset += N
    # Initialize CSP
    pitch_contour = PitchContour()
    pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
    pitch_contour.setNotes(N, K, probabilities, bins)
    solution = pitch_contour.solve()
    # TODO: add evaluation code here 
