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
	out_arr = output.data[0].cpu().numpy()
	tar_arr = target.data[0].cpu().numpy()
	out_pitch = np.argmax(out_arr , axis=1)
	tar_pitch = np.argmax(tar_arr, axis = 1)
	data_pitch = np.argmax(data.data[0].cpu().numpy())
	print(out_pitch, tar_pitch, data_pitch)
	#acc,cnt = eval_accuracy(out_pitch, tar_pitch,len(train_loader))
	#print("accuracy:", float(acc/len(out_pitch)))
	acc = sum([int(abs(a-b)<10) for a,b in zip(out_pitch, tar_pitch)])
	accuracies.append(float(acc/len(out_pitch)))
	acc2 = sum([int(abs(a-b)<10) for a,b in zip(data_pitch, tar_pitch)])
	print("accuracy:", float(acc/len(out_pitch)), float(acc/len(out_pitch)])
	if idx>5:
		break
print(accuracies)
print(np.max(np.asarray(accuracies)))
