import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet
import sys

debug = False

def getNumberOfHits(ground_truth, prediction, N):
    numCorrect = 0
    for i in range(N):
          if ground_truth[i] == prediction[i]:
            numCorrect +=1
    return numCorrect

annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = PitchEstimationDataSet(annotations_val, '/root/data/val/')

# Load CNN results from validation set
filepath = "dataset/val_result_mtrx.npy"
validation_inference = np.load(filepath)

# Load trained Maximum Likelihood Estimates
trained_mle = "dataset/transitions_mle.npy" # Path to saves parameters for HMM
transitions = np.load(trained_mle).item()

# Save probabilities that give best results for HMM and worst result for HMM
K=5
b_prob = np.zeros((1,K))
w_prob = np.zeros((1,K))

# Run inference on each song
hmmTotalAccuracy = []
# rangeM = [20, 50, 100, 200, 300, 500, 1000]
# for M in [50, 100, 200, 300, 400, 500]:
rangeM = [300]
for M in rangeM:
  totalAccuracy = 0
  cnnOnlyAccuracy = 0
  offset = 0

  for songID in range(1):
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
          pitch_contour = PitchContour()
          pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
          pitch_contour.setStartProbability(lambda v : transitions[(lastBin, v)])
          # print ("Setting notes for the CSP")
          pitch_contour.setNotes(M, K, probabilities[i * M:(i + 1) * M, :], bins[i * M:(i + 1) * M, :])
          # print ("Solving CSP...")
          solution = pitch_contour.solve()
          lastBin = solution[M-1]
          currentAccuracy = getNumberOfHits(val_set.pitches[songID][i*M:(i+1)* M], solution, M)
          currentCnnOnlyAccuracy = getNumberOfHits(val_set.pitches[songID][i*M:(i+1)* M], bins[:, 0][i*M:(i+1)* M], M)
          # print ("With HMM: Accuracy rate on this song %f " % (currentAccuracy/M))
          # print ("Without HMM: Accuracy rate on this song %f " % (currentCnnOnlyAccuracy/M))
          if currentAccuracy / M > currentCnnOnlyAccuracy / M + 0.1:
              np.append(b_prob, probabilities[i * M:(i + 1) * M, :], axis = 0)
          if currentAccuracy / M < currentCnnOnlyAccuracy / M - 0.1:
              np.append(w_prob, probabilities[i * M:(i + 1) * M, :], axis = 0)
          cnnOnlyAccuracy += currentCnnOnlyAccuracy
          totalAccuracy += currentAccuracy


      if remainder > 0:
          # print ('Fragment %d to %d' % (patches * M, N))
          pitch_contour = PitchContour()
          pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
          pitch_contour.setStartProbability(lambda v : transitions[(lastBin, v)])
          # print ("Setting notes for the CSP")
          pitch_contour.setNotes(remainder, K, probabilities[patches*M:N, :], bins[patches*M:N, :])
          # print ("Solving CSP...")
          solution = pitch_contour.solve()
          currentAccuracy = getNumberOfHits(val_set.pitches[songID][patches*M:N], solution, remainder)
          currentCnnOnlyAccuracy = getNumberOfHits(val_set.pitches[songID][patches*M:N], bins[:, 0][patches*M:N], remainder)
          print ("With HMM: Accuracy rate on this song %f " % (currentAccuracy/remainder))
          print ("Without HMM: Accuracy rate on this song %f " % (currentCnnOnlyAccuracy/remainder))
          if currentAccuracy / remainder > currentCnnOnlyAccuracy / remainder + 0.1:
              np.append(b_prob, probabilities[patches*M:N, :], axis = 0)
          if currentAccuracy / remainder < currentCnnOnlyAccuracy / remainder - 0.1:
              np.append(w_prob, probabilities[patches*M:N, :], axis = 0)
          cnnOnlyAccuracy += currentCnnOnlyAccuracy
          totalAccuracy += currentAccuracy
      sys.stdout.flush()
  hmmTotalAccuracy.append(totalAccuracy/val_set.lengths[-1])



print (rangeM, hmmTotalAccuracy)
# print ("With HMM: Total accuracy rate")
# print (totalAccuracy/val_set.lengths[-1])
print ("Without HMM: Total accuracy rate")
print (cnnOnlyAccuracy/val_set.lengths[-1])
print (b_prob)
np.save("good_prob_hmm", b_prob)
np.save("bad_prob_hmm", w_prob)
