from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader 
from model_lstm import *
from LSTMDataset import *
import sys
from time import time
from util import *
import numpy as np

id  = 0
model = torch.load('lstm_model/lstm_epoch0.pt')

annotations_train = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/'
train_set = LSTMDataSet(annotations_train, '/root/CS229-MusicTranscription/dataset/train_lstm_input.npy')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

annotations_val = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
val_set = LSTMDataSet(annotations_val, '/root/CS229-MusicTranscription/dataset/val_lstm_input.npy')
val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

accuracies = []
for idx, dictionary in enumerate(val_loader):
	data, target = dictionary['input'], dictionary['freq_vec']
	data = torch.transpose(data, 1,0)
	target = torch.transpose(target, 1,0)
	data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.FloatTensor)
	data, target = data.cuda(), target.cuda()
	output = model(data)
	print("!!",len(val_loader))
	acc,cnt = eval_accuracy(output.data[0].cpu().numpy(), target.data[0].cpu().numpy(),len(train_loader))
	print("accuracy:", acc/cnt)
	accuracies.append(acc)
print(accuracies)
print(np.mean(np.asarray(accuracies)))
