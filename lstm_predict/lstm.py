#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to('cpu'))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to('cpu'))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)
        out = self.dropout(out)

        return out
