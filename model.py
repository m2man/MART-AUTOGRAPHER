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
    def __init__(self, classCount=20, structure='b4', freeze=True, dropout=None):
        super(EfficientNet_MART, self).__init__()
        # Init model
        self.classCount = classCount
        self.structure = structure
        self.dropout = dropout
        full_structure = 'efficientnet-' + self.structure
        self.backbone = EfficientNet.from_pretrained(full_structure)
        
        # freeze backbone
        if freeze:
            for n, p in self.backbone.named_parameters():
                p.requires_grad = False
                
        # Final linear layer
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if structure == 'b4':
            features_shape = 1792
        if structure == 'b0':
            features_shape = 1280
        self.fc = nn.Linear(features_shape, 1048)
        self.fc2 = nn.Linear(1048, self.classCount)
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
        x = F.relu(x)
        if self.dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        
        return x