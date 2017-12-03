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
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(109, self.hidden_dim)

        # The linear layer that maps from hidden state space to notes (pitch bins)
        self.fc = nn.Linear(self.hidden_dim, 109)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

    def forward(self, input_prob):
        lstm_out, self.hidden = self.lstm(
            input_prob.view(109, 1, -1), self.hidden)
        notes = self.fc(lstm_out.view(109, -1))
        return F.log_softmax(notes)
