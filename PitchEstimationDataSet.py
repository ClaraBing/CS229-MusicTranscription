from __future__ import print_function, division
from scipy import misc
from scipy import ndimage
import os
import torch
import numpy as np
from torchvision import transforms, utils
from util import read_melody_avg, read_melody
from torch.utils.data import Dataset
from collections import Counter
import sys

class PitchEstimationDataSet(Dataset):
    """Pitch Estimation dataset."""

    def __init__(self, annotations_dir, images_dir, sr_ratio=6, audio_type='RAW', multiple=False, fusion_mode='no_fusion', transform=None):
        """
        Args:
            annotations_dir (string): Path to the annotation folder that contains
            the CSVs.
            images_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = images_dir
        self.audio_type = audio_type
        self.multiple = multiple
        self.fusion_mode = fusion_mode
        self.transform = transform
        # Load all CSVs and count number of frames in the total dataset
        self.pitches = []
        self.hasNote = []
        self.lengths = [] #Cumulated lengths
        self.numberOfSongs = len(os.listdir(annotations_dir))
        self.songNames = []
        self.rawNames = []
        self.songLengths = [] # Length of a song
        self.currentCount = 0
        self.sr_ratio = sr_ratio
        invalid_path_cnt = 0
        for filename in os.listdir(annotations_dir):
            if filename.endswith(".csv"):
                # The ordering/lengths of songs can be determined following the code below:
                audioName = filename[:filename.find('MELODY')-1] # remove the trailing '_MELODY1.csv'
                self.songNames.append(audioName)
                if self.multiple:
                    new_bin_melody, _ = read_melody(audioName, annotations_dir, sr_ratio=self.sr_ratio, multiple=self.multiple)
                else:
                    new_bin_melody, _, new_hasNote = read_melody_avg(audioName, annotations_dir, sr_ratio=2*self.sr_ratio, multiple=self.multiple) # len(new_bin_melody) denotes the length of a song
                    self.hasNote.append(new_hasNote)
                # sanity check: if not enough images (e.g. may differ by ~10), chop the gt pitches
                    for pid in range(len(new_bin_melody)):
                        img_name = os.path.join(self.images_dir, audioName+"/spec_"+audioName+"_{:s}_".format(self.audio_type)+str(pid)+".png")
                        if not os.path.exists(img_name):
                            print('Chopped {:s} to {:d} ({:d} shorted)'.format(img_name, pid, len(new_bin_melody)-pid))
                            new_bin_melody = new_bin_melody[:pid]
                            invalid_path_cnt += 1
                            break
                self.lengths.append(len(new_bin_melody)+ self.currentCount)
                self.songLengths.append(len(new_bin_melody))
                self.currentCount += len(new_bin_melody)
                self.pitches.append(new_bin_melody)

        print('Class count from PitchEstimationDataSet (total={:d} / invalid={:d}):'.format(sum([len(pitches) for pitches in self.pitches]), invalid_path_cnt))
        if not self.multiple:
            print(Counter([p for pitches in self.pitches for p in pitches]))


    def __len__(self):
        return self.currentCount


    def __getitem__(self, idx, aug_noise=False, aug_vol=False):
        # Find which song the annotated time frame belongs to:
        if len(self.lengths) == 1:
          songId = 0
        else:
          songId = next(x[0] for x in enumerate(self.lengths) if x[1] > idx)
        songName = self.songNames[songId]
        pitchId = idx if songId == 0 else idx - self.lengths[songId - 1]

        # example img path: '.../train/MusicDelta_FusionJazz/spec_MusicDelta_FusionJazz_RAW_986.png', i.e. 'RAW_{img_id}' rather than 'RAW_{track_id}_{img_id}'
        img_name = os.path.join(self.images_dir, songName + "/spec_"+ songName+"_{:s}_".format(self.audio_type)+str(pitchId)+".png")

        image = np.transpose(ndimage.imread(img_name, mode='RGB'), (2,0,1)) # np.transpose: change from H*W*C to C*H*W
        frequency = torch.Tensor(self.pitches[songId][pitchId]) if self.multiple else self.pitches[songId][pitchId]
        # hasNote = torch.Tensor(self.hasNote[songId][pitchId]) if self.multiple else self.hasNote[songId][pitchId]
        hasNote = torch.Tensor([1 if p else 0 for p in self.pitches[songId][pitchId]]) if self.multiple else (1 if self.pitches[songId][pitchId]>0 else 0)
        # prepare different sample format for fusion mode
        if self.fusion_mode == 'no_fusion':
            sample = {'image': image, 'frequency': frequency, 'hasNote':hasNote}# , 'song':songName, 'image_path':img_name, 'idx':idx}
        else:
            cqt_img_name = img_name.replace('image', 'cqt_image')
            cqt_image = np.transpose(ndimage.imread(cqt_img_name, mode='RGB'), (2,0,1))
            if self.fusion_mode == 'stacking':
                image = np.concatenate((image, cqt_image), axis=0)
                sample = {'image': image, 'frequency': frequency}# , 'song':songName, 'image_path':img_name, 'idx':idx}
            elif self.fusion_mode == 'early_fusion' or self.fusion_mode == 'late_fusion':
                sample = {'mel':image, 'cqt': cqt_image, 'frequency': frequency}

        if self.transform and self.fusion_mode == 'no_fusion':
            # NOTE: currently transform is only supported for no fusion mode
            sample = self.transform(sample)

        return sample

