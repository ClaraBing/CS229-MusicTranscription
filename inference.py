## Runs inference on some audio file
from util import *
from model import Net
from pitch_contour import *
from scipy import ndimage
from fragmenter_hmm_solver import fragmented_solver
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
cnn_weights = "output_model_old/conv5_13/model_conv5_train_epoch9.pt" # Path to saved weights of CNN
trained_mle = "dataset/transitions_mle.npy" # Path to saves parameters for HMM
saved_probabilities = "probabilities_"+filename
saved_bins = "bins_"+filename

hasGroundTruth = True
annotationsDirectory = "/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/"
if hasGroundTruth:
  ground_truth, _ = read_melody(filename, dir=annotationsDirectory)
  outputMIDI(len(ground_truth), ground_truth, filename+'_result',  duration_sec = 0.058)
  print ("Saved %d notes" % len(ground_truth))

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
N = len(os.listdir(imageOutputDirectory)) # Number of pitches
bins = np.zeros((N,K))
probabilities = np.zeros((N,K))

# Naively perform inference on each of the timeframes and save results
cnn_inf = True
if cnn_inf:
  print ("Pitch estimation on each frame .")
  for i in range(N):
      print(("%d out of %d") % (i, N-1))
      path = '{:s}spec_{:s}_{:d}.png'.format(imageOutputDirectory, filename+"_MIX", i)
      # print (path)
      image = torch.from_numpy(np.transpose(ndimage.imread(path, mode='RGB'), (2,0,1)))
      data = Variable(image).type(torch.FloatTensor)
      data = data.unsqueeze(0)
      output = trained_cnn(data).data.numpy()
      bin = np.argsort(-output)[0][:K].reshape((1,K))
      bins[i] = bin
      probabilities[i] = np.exp(output[0][bin])
      probabilities[i] /= np.sum(probabilities[i])
  print("Done.")
  np.save(saved_probabilities, probabilities)
  np.save(saved_bins, bins)
else:
  probabilities = np.load(saved_probabilities+".npy")
  bins = np.load(saved_bins+".npy")


# Initialize Pitch Contour
print ("Pitch tracking on %d notes" % N)
transitions = np.load(trained_mle).item()
solution = fragmented_solver(N, K, 300, probabilities, bins, transitions, threshold=0.6)
print ("Done.")

# Output MIDI file
print ("Save as MIDI file.")
bins = [solution[i] for i in range(N)]
outputMIDI(N, bins, filename+'_result',  duration_sec = 0.058)
print ("Done. The file was saved at midiOutput/"+filename+"_result.mid")
