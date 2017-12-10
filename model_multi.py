# utils
from __future__ import print_function
import argparse
from time import time
# torch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
# our code
# from PitchEstimationDataSet import *

class Net_Multi(nn.Module):
    def __init__(self):
        super(Net_Multi, self).__init__()
        # conv: 3*256*256   -->  64*256*256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.conv1_bn = nn.BatchNorm2d(64)
        # max pool to 64*128*128
        # conv: 64*128*128  -->  64*128*128
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Nov 25: Change Dropout to Batch normalization
        self.conv2_bn = nn.BatchNorm2d(64) # used to be: self.conv2_drop = nn.Dropout2d()
        # max pool to 64*64*64
        # conv: 64*64*64  -->  96*64*64
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(96)
        # max pool to 96*32*32
        # conv: 96*32*32  -->  64*32*32
        self.conv4 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        # conv: 64*32*32  -->  32*32*32
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(32)
        # max pool to 32*16*16
        # reshape(flatten) to 8192
        self.fc1 = nn.Linear(8192, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 108)

    def forward(self, x):
        # print('in forward:')
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        # print(x.data.shape)
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(F.max_pool2d(self.conv5_bn(self.conv5(x)), 2))
        # print(x.data.shape)
        x = x.view(-1, 8192)
        # print(x.data.shape)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.fc2(x)
        return x # F.log_softmax(x)
    
    def get_features(self, x):
        # print('in forward:')
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        # print(x.data.shape)
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(F.max_pool2d(self.conv5_bn(self.conv5(x)), 2))
        # print(x.data.shape)
        return x.view(-1, 8192)

    def print_archi(self):
        print('Conv1: 3  --> 64, kernal=7*7')
        print('Max pool: 2*2')
        print('Conv2: 64 --> 64, kernal=3*3')
        print('Max pool: 2*2')
        print('Conv3: 64 --> 96, kernal=3*3')
        print('Max pool: 2*2')
        print('Conv4: 96 --> 64, kernal=3*3')
        print('Conv5: 64 --> 32, kernal=3*3')
        print('Max pool: 2*2')
        print('FC1: 8192 --> 2048')
        print('FC2: 2048 --> 108')
