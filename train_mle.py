import numpy as np
from util import read_melody
from pitch_contour import trainTransition
import os 
from config import *

configuration = config_cqt()
# load training annotations
annotations_dir = configuration['annot_folder']+'train/'

melodies = []
print ("Loading training data ...")
for filename in os.listdir(annotations_dir):
    if filename.endswith(".csv"):
        audioName = filename[:filename.find('MELODY')-1] # remove the trailing '_MELODY1.csv'
        new_bin_melody, _ = read_melody(audioName, dir=annotations_dir, sr_ratio=2 * configuration['sr_ratio'])
        melodies.append(new_bin_melody)
        print ("Loaded annotations for "+ audioName)

bins = range(109)

print ("Finished loading data")

# Maximum Likelihood estimates
# store in outputFile
outputFile = 'transitions_mle_'+str(configuration['sr_ratio'])
print ("Training transitions ...")
trainTransition(melodies, bins, outputFile, alpha=1)

print("Done.")
print("Verifying transitions ...")
# Verify soundness of transition estimates
transitions = np.load(outputFile+".npy").item()
for i in range(109):
  count = 0
  for j in range(109):
    count += transitions[(i,j)]
  if (int(round(count)) != 1):
    print(count)

print ("Done. Transitions were saved in "+ outputFile)
