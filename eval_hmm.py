import numpy as np
from pitch_contour import *

# annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
# val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/')
# val_loader = DataLoader(val_set, shuffle=False, **kwargs)

# Load CNN results from validation set
filepath = "val.npy"
validation_inference = np.load(filepath)

print (validation_inference.shape)
# Get probabilities and bins for the first N notes
N = 300
K = 5
probabilities = np.zeros((N, K))
bins = np.zeros((N,K))
print ()
for i in range(N):
    probabilities[i] = validation_inference[0][:K][:,0]
    bins[i] = validation_inference[i][:K][:,1]
    pass

# Initialize CSP
trained_mle = "transitions_mle.npy" # Path to saves parameters for HMM
transitions = np.load(trained_mle).item()
pitch_contour = PitchContour()
pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
pitch_contour.setNotes(N, K, probabilities, bins)
solution = pitch_contour.solve(mode="gibbs")
print (solution)
