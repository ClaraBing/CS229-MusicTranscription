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

class Net_Conv7(nn.Module):
    def __init__(self):
        super(Net_Conv7, self).__init__()
        # conv: 3*256*256   -->  64*256*256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        # max pool to 64*128*128
        # conv: 64*128*128  -->  128*128*128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128) # used to be: self.conv2_drop = nn.Dropout2d()
        # conv: 128*128*128  -->  128*128*128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        # max pool to 128*64*64
        # conv: 128*64*64  -->  256*64*64
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        # max pool to 256*32*32
        # conv: 256*32*32  -->  256*32*32
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        # max pool to 256*16*16
        # conv: 256*16*16  -->  512*16*16
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(512)
        # max pool to 512*8*8
        # conv: 512*8*8  -->  512*8*8
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(512)
        # max pool to 512*4*4
        # reshape(flatten) to 8192
        self.fc1 = nn.Linear(8192, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 109)

    def forward(self, x):
        # print('in forward:')
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        # print(x.data.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(F.max_pool2d(self.conv3_bn(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.conv4_bn(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.conv5_bn(self.conv5(x)), 2))
        x = F.relu(F.max_pool2d(self.conv6_bn(self.conv6(x)), 2))
        x = F.relu(F.max_pool2d(self.conv7_bn(self.conv7(x)), 2))
        # print(x.data.shape)
        x = x.view(-1, 8192)
        # print(x.data.shape)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


    def print_archi(self):
        print('Conv1: 3   --> 64, kernal=3*3')
        print('Max pool: 2*2')
        print('Conv2: 64  --> 128, kernal=3*3')
        print('Conv3: 128 --> 128, kernal=3*3')
        print('Max pool: 2*2')
        print('Conv4: 128 --> 256, kernal=3*3')
        print('Max pool: 2*2')
        print('Conv5: 256 --> 256, kernal=3*3')
        print('Max pool: 2*2')
        print('Conv6: 256 --> 512, kernal=3*3')
        print('Max pool: 2*2')
        print('Conv7: 512 --> 512, kernal=3*3')
        print('Max pool: 2*2')
        print('FC1: 8192 --> 2048')
        print('FC2: 2048 --> 109')
