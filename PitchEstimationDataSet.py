from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
        for filename in os.listdir(annotations_dir):
            if filename.endswith(".csv"):
                self.songNames.append(filename[:-4])
                new_melody = read_melody(filename[:-4])
                self.lengths.append(len(new_melody)+ self.lengths[-1])
                self.pitches.append(new_melody)

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, idx):
        # Find which song the annotated time frame belongs to:
        songId = next(x[0] for x in enumerate(self.lengths) if x[1] < idx)
        songName = self.songNames[songId]
        pitchId = idx if songName == 0 else idx - self.lengths[songId - 1]
        img_name = os.path.join(self.images_dir, "spec_"+ songName+"_MIX_"+pitchId+".png")
        image = io.imread(img_name)
        sample = {'image': image, 'frequency': self.pitches[songId][pitchId]}

        if self.transform:
            sample = self.transform(sample)

        return sample
