import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMMultiNotes(nn.Module):

    def __init__(self):
        super(LSTMMultiNotes, self).__init__()
        self.hidden_dim = 1024
        self.num_layers = 2

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, nlayers)
        self.lstm = nn.LSTM(109, self.hidden_dim, 2)

        # The linear layer that maps from hidden state space to notes (pitch bins)
        self.fc = nn.Linear(self.hidden_dim, 109)
        self.init_weights()
        self.hidden = self.init_hidden()

    def init_weights(self):
        initrange = 0.1
        # self.lstm.weight.data.uniform(-initrange, initrange)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)).cuda())

    def forward(self, input_prob):
        lstm_out, self.hidden = self.lstm(
            input_prob.view(input_prob.size()[0], 1, -1), self.hidden)
        notes = self.fc(lstm_out.view(input_prob.size()[0], -1))
        return F.log_softmax(notes)
