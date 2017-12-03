import numpy as np
from pitch_contour import PitchContour, transition_probability
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
filepath = "val.npy"
validation_inference = np.load(filepath)

# Run inference on each song
K = 5
hmmTotalAccuracy = {}
rangeMu = [0.01, 0.1, 0.4, 0.9, 1, 1.5]
rangeSigma = [1.0, 2.4, 3.0, 5.0, 10]
#rangeM = [20, 50, 100, 200, 300, 500, 1000]
# for M in [50, 100, 200, 300, 400, 500]:
M = 100
for mu in rangeMu:
  for sigma in rangeSigma:
      totalAccuracy = 0
      cnnOnlyAccuracy = 0
      offset = 0

      for songID in range(val_set.numberOfSongs):
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
          lastBin = 0
          sys.stdout.flush()
          for i in range(patches):
              # print ('Fragment %d to %d' % (i * M, (i + 1) * M))
              pitch_contour = PitchContour(mu = mu, sigma = sigma)
              pitch_contour.setStartProbability(lambda v : transition_probability(lastBin, v, mu, sigma))
              # print ("Setting notes for the CSP")
              pitch_contour.setNotes(M, K, probabilities[i * M:(i + 1) * M, :], bins[i * M:(i + 1) * M, :])
              # print ("Solving CSP...")
              solution = pitch_contour.solve()
              lastBin = solution[M-1]
              currentAccuracy = getNumberOfHits(val_set.pitches[songID][i*M:(i+1)* M], solution, M)
              currentCnnOnlyAccuracy = getNumberOfHits(val_set.pitches[songID][i*M:(i+1)* M], bins[:, 0][i*M:(i+1)* M], M)
              print ("With HMM: Accuracy rate on this song %f " % (currentAccuracy/M))
              print ("Without HMM: Accuracy rate on this song %f " % (currentCnnOnlyAccuracy/M))
              cnnOnlyAccuracy += currentCnnOnlyAccuracy
              totalAccuracy += currentAccuracy

          if remainder > 0:
              # print ('Fragment %d to %d' % (patches * M, N))
              pitch_contour = PitchContour()
              # pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
              pitch_contour.setStartProbability(lambda v : transition_probability(lastBin, v, mu, sigma))
              # print ("Setting notes for the CSP")
              pitch_contour.setNotes(remainder, K, probabilities[patches*M:N, :], bins[patches*M:N, :])
              # print ("Solving CSP...")
              solution = pitch_contour.solve()
              currentAccuracy = getNumberOfHits(val_set.pitches[songID][patches*M:N], solution, remainder)
              currentCnnOnlyAccuracy = getNumberOfHits(val_set.pitches[songID][patches*M:N], bins[:, 0][patches*M:N], remainder)
              print ("With HMM: Accuracy rate on this song %f " % (currentAccuracy/remainder))
              print ("Without HMM: Accuracy rate on this song %f " % (currentCnnOnlyAccuracy/remainder))
              cnnOnlyAccuracy += currentCnnOnlyAccuracy
              totalAccuracy += currentAccuracy
          sys.stdout.flush()
      hmmTotalAccuracy[(mu, sigma)] = (totalAccuracy/val_set.lengths[-1])
      print ((mu, sigma), hmmTotalAccuracy[(mu, sigma)])



for key, v in hmmTotalAccuracy.items():
    print (key,v)
# print ("With HMM: Total accuracy rate")
# print (totalAccuracy/val_set.lengths[-1])
print ("Without HMM: Total accuracy rate")
print (cnnOnlyAccuracy/val_set.lengths[-1])
