# Ref: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
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
import os
from PitchEstimationDataSet import PitchEstimationDataSet

#CUDA_VISIBLE_DEVICES=1
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# TODO: command line args + kwargs
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

# data
annotations_train = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/'
train_set = LSTMDataSet(annotations_train, '/root/CS229-MusicTranscription/dataset/train_lstm_input.npy')
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

# model
model = LSTMMultiNotes(1024, 2, 32)
model.cuda()

val_set = PitchEstimationDataSet(annotations_train, '/root/data/val/')

loss_function = nn.NLLLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters())

def train(model, train_loader, criterion, num_epoch):
    model.train()
    for epoch in range(num_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        for idx, dictionary in enumerate(train_loader):
#            print(idx)
            batch_start = time()
            data, target = dictionary['input'], dictionary['freq_vec']
            data = torch.transpose(data, 1,0)
            target = torch.transpose(target, 1,0)
            data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.FloatTensor)
            data, target = data.cuda(), target.cuda()

            model.zero_grad()
            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()
    
            # Step 3. Run our forward pass.
            output = model(data)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print('Train Epoch: epoch {} iter {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime per batch: {:.6f}s'.format(epoch, idx, idx*len(data), len(train_loader.dataset), 100.*idx/len(train_loader.dataset), loss.data[0], time()-batch_start))
                sys.stdout.flush()
            if idx % 200 == 0:
           #     print("The accuracy is :", eval_accuracy(output, target,idx))
                torch.save(model, 'lstm_model/lstm_epoch{:d}.pt'.format(idx))

# See what the scores are after training
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# tag_scores = model(inputs)

if __name__ == '__main__':
    criterion = nn.MultiLabelSoftMarginLoss()
    train(model, train_loader, criterion, 10)

