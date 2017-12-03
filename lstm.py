# Ref: http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader 
from model_lstm import *

# TODO: command line args + kwargs

# data
annotations_train = '/root/MedleyDB_selected/Annotations/Melody_Annotations/MELODY1/train/'
train_set = LSTMDataSet(annotations_train, '/root/CS229-MusicTranscription/dataset/train_lstm_input.npy')
train_loader = DataLoader(trainint_set, batch_size=1, shuffle=True)

# model
model = LSTMMultiNotes()
model.cuda()

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, criterion, num_epoch):
    model.train()
    for epoch in range(num_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        for idx, dictionary in enumerate(train_loader):
            data, target = dictionary['input'], dictionary['freq_vec']
            data, target = Variable(data).type(torch.FloatTensor), Variable(target).type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            model.zero_grad()
            # Clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()
    
            # Step 3. Run our forward pass.
            output = model(sentence_in)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(tag_scores, targets)
            loss.backward()
            optimizer.step()

# See what the scores are after training
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# tag_scores = model(inputs)

if __name__ == '__main__':
    criterion = nn.MultiLabelSoftMarginLoss()
    train(model, train_loader, criterion, 10)
