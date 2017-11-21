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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv: 3*256*256   -->  64*256*256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        # max pool to 64*128*128
        # conv: 64*128*128  -->  64*128*128
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_drop = nn.Dropout2d()
        # max pool to 64*64*64
        # conv: 64*64*64  -->  96*64*64
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        # max pool to 96*32*32
        # conv: 96*32*32  -->  64*32*32
        self.conv4 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        # conv: 64*32*32  -->  32*32*32
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # max pool to 32*16*16
        # reshape(flatten) to 8192
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, 109)

    def forward(self, x):
        # print('in forward:')
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.data.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        # print(x.data.shape)
        x = x.view(-1, 8192)
        # print(x.data.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


