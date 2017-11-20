from __future__ import print_function, division
from scipy import misc
from scipy import ndimage
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from read_melody import *

class PitchEstimationDataSet(Dataset):
    """Pitch Estimation dataset."""

    def __init__(self, annotations_dir, images_dir, transform=None):
        """
        Args:
            annotations_dir (string): Path to the annotation folder that contains
            the CSVs.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.transform = transform
        # Load all CSVs and count number of frames in the total dataset
        self.pitches = []
        self.lengths = []
        self.numberOfSongs = len(os.listdir(annotations_dir))
        self.songNames = []
        self.currentCount = 0
        for filename in os.listdir(annotations_dir):
            if filename.endswith(".csv"):
                audioName = filename[:filename.find('MELODY')-1] # remove the trailing '_MELODY1.csv'
                self.songNames.append(audioName)
                new_melody = read_melody(audioName, annotations_dir)
                self.lengths.append(len(new_melody)+ self.currentCount)
                self.currentCount += len(new_melody)
                self.pitches.append(new_melody)
                # print (self.currentCount)
    def __len__(self):
        # print (self.currentCount)
        return self.currentCount

    def __getitem__(self, idx):
        # Find which song the annotated time frame belongs to:
        if len(self.lengths) == 1:
          songId = 0
        else:
          songId = next(x[0] for x in enumerate(self.lengths) if x[1] < idx)
        songName = self.songNames[songId]
        pitchId = idx if songId == 0 else idx - self.lengths[songId - 1]
        img_name = os.path.join(self.images_dir, songName + "/spec_"+ songName+"_MIX_"+str(pitchId)+".png")
        # np.transpose: change from H*W*C to C*H*W
        image = np.transpose(ndimage.imread(img_name, mode='RGB'), (2,0,1))
        sample = {'image': image, 'frequency': self.pitches[songId][pitchId]}

        if self.transform:
            sample = self.transform(sample)

        return sample

# path_to_annotations = '../MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/'
# path_to_images = '../data/'
# raw_dataset = PitchEstimationDataSet(path_to_annotations, path_to_annotations)
