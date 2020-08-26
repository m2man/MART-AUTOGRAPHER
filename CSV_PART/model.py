import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device('cuda')

class Tabular_Model(nn.Module):
    def __init__(self, input_dim=102, output_dim=20, layer_dim=[1000, 500], dropout=0.5, batch_norm=True):
        super(Tabular_Model, self).__init__()
        # Init model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        if isinstance(layer_dim, list):
            self.layer_dim = layer_dim
        else:
            self.layer_dim = [layer_dim]
             
        if dropout is not None:
            self.DropoutList = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.layer_dim))])
            
        if batch_norm:
            self.BatchnormList = nn.ModuleList([nn.BatchNorm1d(num_features=x) for x in self.layer_dim])
            self.firstbn = nn.BatchNorm1d(num_features=self.input_dim)
        
        self.LinearList = nn.ModuleList([nn.Linear(self.input_dim, self.layer_dim[0])])
        self.LinearList.extend([nn.Linear(self.layer_dim[x], self.layer_dim[x+1]) for x in range(0, len(self.layer_dim)-1)])
        self.classifier = nn.Linear(self.layer_dim[-1], self.output_dim)

  
    def forward(self, inputs):
        x = self.extract_features(inputs=inputs, train=True)
        x = self.classifier(x)
        return x

    def extract_features(self, inputs, train=True):
        if train:
            x = self.firstbn(inputs)
        else:
            x = inputs
        for step in range(len(self.layer_dim)-1):
            x = self.LinearList[step](x)
            if self.batch_norm is not None and train:
                x = self.BatchnormList[step](x)
            x = F.relu(x)
            if self.dropout is not None and train:
                x = self.DropoutList[step](x)
        
        x = self.LinearList[-1](x)
        if self.batch_norm is not None and train:
            x = self.BatchnormList[-1](x)
        if train:
            x = F.relu(x)
        if self.dropout is not None and train:
            x = self.DropoutList[-1](x)
        return x