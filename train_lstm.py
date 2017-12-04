import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from LSTMDataset import *
from util import *

torch.manual_seed(1)

n_features = 109
n_classes = 109
n_layers = 2
n_epochs = 5
n_batch = 9
batch_size = 32

model =  nn.LSTM(109,1024,2)
model.cuda()

optimizer = optim.Adam(model.parameters())

kwargs = {}

# train
annotations_train = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/val/'
train_set = LSTMDataSet(annotations_train, '/root/CS229-MusicTranscription/dataset/val_lstm_input.npy')
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

model.train()
errors = []
for epoch in range(n_epochs):
    training_loss = 0
    for idx, dictionary in enumerate(train_loader):
        print("dic", dictionary['input'])
        batch_start = time()
        data, target = dictionary['input'], dictionary['freq_vec']
        print("data, target",data, target)
        data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.FloatTensor)
        data, target = data.cuda(), target.cuda()
        
        model.zero_grad()
        #x = np.reshape(Xtrain_lst[i], (1,batch_limit,n_features))
        #data = autograd.Variable(torch.FloatTensor(x))
        #target = autograd.Variable(torch.FloatTensor(Ytrain_lst[i]))
        model.hidden = model.init_hidden()
        output = model(data)
        #last_output = output[-1]
        loss = nn.MultiLabelSoftMarginLoss()
        err = loss(last_output, target)
        training_loss+=err.data[0]
#         if (len(errors)>100):
#             break
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: epoch {} iter {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime per batch: {:.6f}s'.format(epoch, idx, idx*len(data), len(train_loader.dataset), 100.*idx/len(train_loader.dataset), loss.data[0], time()-batch_start))
            sys.stdout.flush()
    torch.save(model, 'lstm2_epoch{:d}.pt'.format(idx+1))
    #errors.append(training_loss/len(Xtrain_lst))
    
training_loss = training_loss/len(Xtrain_lst)
print("training loss:" +str(training_loss))
