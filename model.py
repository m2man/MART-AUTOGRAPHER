import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torchvision
from torchsummary import summary

from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4')

device = torch.device('cuda')

class EfficientNet_MART(nn.Module):
    def __init__(self, classCount=20, structure='b4', freeze=True, dropout=None, hidden_size=1024, batch_norm=False):
        super(EfficientNet_MART, self).__init__()
        # Init model
        self.classCount = classCount
        self.structure = structure
        self.dropout = dropout
        full_structure = 'efficientnet-' + self.structure
        self.backbone = EfficientNet.from_pretrained(full_structure)
        self.hidden_size = hidden_size
        
        # freeze backbone
        if freeze:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(num_features=self.hidden_size)
        else:
            self.bn = None
            
        # Final linear layer
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if structure == 'b4':
            features_shape = 1792
        if structure == 'b0':
            features_shape = 1280
        self.fc = nn.Linear(features_shape, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.classCount)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, inputs):
        #bs = inputs.size(0)

        # Convolution layers
        x = self.backbone.extract_features(inputs)

        # Pooling and final linear layer
        x = self.avg_pooling(x)
        #x = x.view(bs, -1)
        x = x.flatten(start_dim=1)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        
        return x