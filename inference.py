## Runs inference on some audio file
from util import *
from model import Net
from pitch_contour import *
# torch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

directory = "" # Directory containing audio file for inference
filename = "" # Name of audio file for inference
extension = "" # Extension of audio file
imageOutputDirectory = "" # Directory containing spectogram images
cnn_weights = "" # Path to saved weights of CNN
trained_mle = "" # Path to saves parameters for HMM


# Convert audio file to spectogram slices
wav2spec_data(directory, filename, fext, imageOutputDirectory)

# Initialize CNN with pre-trained weights
trained_cnn = Net()
trained_cnn.load_state_dict(torch.load(cnn_weights))

K = 5
bins = []
probabilities = []
N = len(os.listdir(imageOutputDirectory)) # Number of pitches

# Naively perform inference on each of the timeframes and save results
for i in range(N):
    path = '{:s}spec_{:s}_{:d}.png'.format(imageOutputDirectory, filename, i)
    image = np.transpose(ndimage.imread(path, mode='RGB'), (2,0,1))
    data = Variable(image).type(torch.FloatTensor)
    output = model(data).numpy()
    bins = np.argsort(-output)[:K] # only get K first bins
    probabilities.append([np.exp(output[bins])])

# Initialize Pitch Contour
transitions = np.load(trained_mle).item()
pitch_contour = PitchContour()
pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
pitch_contour.setNotes(N, K, probabilities, bins)
solution = pitch_contour.solve()

# Evaluate
# TODO: add evaluation code here

# Output MIDI file
outputMIDI(N, solution, filename+'_result',  duration = 0.3)
