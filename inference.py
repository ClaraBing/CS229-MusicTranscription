## Runs inference on some audio file
from util import *
from model import Net
from pitch_contour import *
from scipy import ndimage
# torch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

directory = "/root/MedleyDB_selected/Audios/" # Directory containing audio file for inference
filename = "MusicDelta_80sRock" # Name of audio file for inference
extension = "wav" # Extension of audio file
imageOutputDirectory = "/root/inference/"+filename+"/" # Directory containing spectogram images
cnn_weights = "output_model/conv5_13/model_conv5_train_epoch9.pt" # Path to saved weights of CNN
trained_mle = "transitions_mle.npy" # Path to saves parameters for HMM

# Convert audio file to spectogram slices
print ("Converting the audio to spectogram images ...")
if not os.path.isdir(imageOutputDirectory):
  os.mkdir(imageOutputDirectory)
  wav2spec_data(directory+filename+"/", filename+"_MIX", extension, imageOutputDirectory)
print ("Done. All files were saved in the folder "+ imageOutputDirectory)
# Initialize CNN with pre-trained weights

trained_cnn = Net()
print ("Load pre-trained CNN ...")
weights = torch.load(cnn_weights)
trained_cnn.load_state_dict(weights['state_dict'])
print("Done.")

K = 5
bins = []
probabilities = []
N = len(os.listdir(imageOutputDirectory)) # Number of pitches

# Naively perform inference on each of the timeframes and save results
print ("Pitch estimation on each frame .")
for i in range(N):
    print(".")
    path = '{:s}spec_{:s}_{:d}.png'.format(imageOutputDirectory, filename+"_MIX", i)
    image = torch.from_numpy(np.transpose(ndimage.imread(path, mode='RGB'), (2,0,1)))
    data = Variable(image).type(torch.FloatTensor)
    output = trained_cnn(data).numpy()
    bins = np.argsort(-output)[:K] # only get K first bins
    probabilities.append([np.exp(output[bins])])
print("Done.")

# Initialize Pitch Contour
print ("Pitch tracking...")
transitions = np.load(trained_mle).item()
pitch_contour = PitchContour()
pitch_contour.setTransitionProbability(lambda b1, b2 : transitions[(b1, b2)])
pitch_contour.setNotes(N, K, probabilities, bins)
solution = pitch_contour.solve()
print ("Done.")

# Evaluate
# TODO: add evaluation code here

# Output MIDI file
print ("Save as MIDI file.")
outputMIDI(N, solution, filename+'_result',  duration = 0.3)
print ("Done. The file was saved at midiOutput/"+filename+"_result")
