import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary

# model = EfficientNet.from_pretrained('efficientnet-b4')

device = torch.device('cuda')

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class MART_Task(nn.Module):
    def __init__(self, classCount=20, input_size=1048, dropout=None, hidden_size=[512, 256], batch_norm=False):
        super(MART_Task, self).__init__()
        # Init model
        self.input_size = input_size
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.classCount = classCount

        if isinstance(hidden_size, list):
            self.hidden_size = hidden_size
        else:
            self.hidden_size = [hidden_size]

        self.SwishList = nn.ModuleList([MemoryEfficientSwish() for _ in range(len(self.hidden_size))])

        if dropout is not None:
            self.DropoutList = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.hidden_size))])

        if batch_norm:
            self.BatchnormList = nn.ModuleList([nn.BatchNorm1d(num_features=x) for x in self.hidden_size])

        self.LinearList = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size[0])])
        self.LinearList.extend([nn.Linear(self.hidden_size[x], self.hidden_size[x+1]) for x in range(0, len(self.hidden_size)-1)])
        self.classifier = nn.Linear(self.hidden_size[-1], self.classCount)

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        for step in range(len(self.hidden_size)):
            x = self.LinearList[step](x)
            if self.batch_norm:
                x = self.BatchnormList[step](x)
            x = self.SwishList[step](x)
            if self.dropout is not None:
                x = self.DropoutList[step](x)
        return x