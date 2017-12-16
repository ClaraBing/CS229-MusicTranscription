import numpy as np
from pitch_contour import PitchContour
from PitchEstimationDataSet import PitchEstimationDataSet
from fragmenter_hmm_solver import fragmented_solver
from config import *
import sys
from time import time
import argparse
'''
This script runs evaluation of the HMM on a given set of songs and prints out
- the accuracy on each song and on the whole set
- stats relevant for comparing HMM with CNN only
Example launch command
python eval_hmm_fragmented --threshold 0.6 --M 0.01
This launches the evaluation on the validation dataset with a threshold of 0.6 and 100 chunks
'''

# Evaluation settings
parser = argparse.ArgumentParser(description='HMM evaluation')
parser.add_argument('--M', type=int,
                    help='Size of chunks for fragmented solver')
parser.add_argument('--threshold', type=float,
                    help='threshold to be used for the HMM solver')
parser.add_argument('--mle', type=str, default='transitions_mle.npy',
                    help='file containing the maximum likelihood parameters')
parser.add_argument('--debug', type=bool, default=False,
                    help='activate debug output')
parser.add_argument('--config', type=str, default='base',
                    help='Configuration to use')
args = parser.parse_args()

if args.config == 'base':
    cfg = config_base()
annotations_val = cfg['annot_folder']
val_set = PitchEstimationDataSet(
    annotations_val,
    cfg['image_folder'],
    sr_ratio=cfg['sr_ratio'],
    audio_type=cfg['audio_type'])

# Load CNN results from validation set
validation_inference = np.load(cfg['saved_weights'])

# Load trained Maximum Likelihood Estimates
trained_mle = "dataset/" + args.mle
transitions = np.load(trained_mle).item()

threshold = args.threshold
M = args.M

totalAccuracy = 0
cnnOnlyAccuracy = 0
totalFix = 0
totalBreak = 0
totalConfirm = 0
offset = 0

print ("Evaluating HMM with parameters M: %f, threshold: %f and config %s" %(M, threshold, args.config))

'''
Method to get the accuracy on a chunk of song and to update general stats.
'''
K=5
b_prob = np.zeros((1, K))
w_prob = np.zeros((1, K))
confirm_prob = np.zeros((1, K)) # Cases where HMM confirms positive result from CNN
fix_prob = np.zeros((1, K)) # Cases where HMM finds positive result when CNN didn't
break_prob = np.zeros((1, K)) # Cases where HMM ignores CNN result and is mistaken

def getNumberOfHits(ground_truth, prediction, N, probs, cnn_prediction = None):
    global b_prob
    global w_prob
    global confirm_prob, fix_prob, break_prob
    numCorrect = 0
    numCNNCorrect = 0
    numConfirm = 0
    numFix = 0
    numBreak = 0
    for i in range(N):
        # Get stats on positive and negative results
        if ground_truth[i] == prediction[i]:
            b_prob = np.append(b_prob, probs[i,:].reshape((1,K)), axis = 0)
            numCorrect +=1
        else:
            w_prob = np.append(w_prob, probs[i,:].reshape((1,K)), axis = 0)
        # Get stats on positive and negative results compared to CNN
        if cnn_prediction is not None:
            if cnn_prediction[i] == ground_truth[i]:
                numCNNCorrect += 1
            if ground_truth[i] == prediction[i] and prediction[i] == cnn_prediction[i]:
                numConfirm +=1
                confirm_prob = np.append(confirm_prob, probs[i, :].reshape((1,K)), axis=0)
            elif ground_truth[i] == prediction[i] and prediction[i] != cnn_prediction[i]:
                numFix +=1
                fix_prob = np.append(fix_prob, probs[i, :].reshape((1,K)), axis=0)
            elif ground_truth[i] != prediction[i] and cnn_prediction[i] == ground_truth[i]:
                numBreak +=1
                break_prob = np.append(break_prob, probs[i, :].reshape((1,K)), axis=0)
    print (numCorrect, numCNNCorrect, numConfirm, numFix, numBreak)
    return numCorrect, numCNNCorrect, numConfirm, numFix, numBreak


for songID in range(len(val_set.songNames)):
    start = time()
    songName = val_set.songNames[songID]
    print ("Evaluating for " + songName)
    N = val_set.songLengths[songID] - 1
    probabilities = np.zeros((N, K))
    bins = np.zeros((N, K))
    print ("Loading for %d notes" % N)
    print (offset, offset + N)
    for i in range(N):
        probabilities[i] = validation_inference[i + offset][:K][:,0]
        probabilities[i] /= np.sum(probabilities[i])
        bins[i] = validation_inference[i + offset][:K][:,1]
    offset += N
    if M == 0:
      fragment = N
    else:
      fragment = M
    solution = fragmented_solver(N, K, fragment, probabilities, bins, transitions, threshold)
    currentAccuracy, currentCnnOnlyAccuracy, confirmAcc, fixAcc, breakAcc \
        = getNumberOfHits(val_set.pitches[songID], solution, N, probabilities, bins[:, 0])
    cnnOnlyAccuracy += currentCnnOnlyAccuracy
    totalAccuracy += currentAccuracy
    totalFix += fixAcc
    totalConfirm += confirmAcc
    totalBreak += breakAcc
    print(M, threshold)
    print ("With HMM: Accuracy rate on this song %f " % (currentAccuracy/N))
    print ("Without HMM accuracy %f" % (currentCnnOnlyAccuracy/N))
    print ("HMM v. CNN stats confirm %f, fix %f, break %f" % (currentAccuracy/N, confirmAcc/N, breakAcc/N))
    print ('Using {:f} seconds'.format(time()-start))
    sys.stdout.flush()

np.save("bad_prob_hmm_refined"+str(threshold), w_prob)
np.save("good_prob_hmm_refined"+str(threshold), b_prob)
np.save("confirm_prob_hmm"+str(threshold), confirm_prob)
np.save("break_prob_hmm"+str(threshold), break_prob)
np.save("fix_prob_hmm"+str(threshold), fix_prob)

print ("With HMM: Total accuracy rate")
print (totalAccuracy/val_set.lengths[-1])
print ('Confirm %f, fix %f, break %f' % (totalAccuracy/N, totalConfirm/N, totalBreak/N))
print ("Without HMM: Total accuracy rate")
print (cnnOnlyAccuracy/val_set.lengths[-1])
